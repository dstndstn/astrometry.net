import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django.core.validators import URLValidator
from django.utils.safestring import mark_safe
from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command


def index(req, user_id):
    submitter = None
    if user_id == None:
        submissions = Submission.objects.all()
    else:
        submitter = get_object_or_404(User, pk=user_id)
    
    context = {'submitter':submitter}
    return render_to_response('submission/by_user.html', context,
        context_instance = RequestContext(req))

class HorizontalRenderer(forms.RadioSelect.renderer):
    def render(self):
        return mark_safe(u'\n'.join([u'%s' % w for w in self]))

class NoBulletsRenderer(forms.RadioSelect.renderer):
    def render(self):
        return mark_safe(u'<br />\n'.join([u'%s' % w for w in self]))

class SubmissionForm(forms.ModelForm):

    file  = forms.FileField(required=False,
                            widget=forms.FileInput(attrs={'size':'80'}))
    url = forms.CharField(widget=forms.TextInput(attrs={'size':'80'}),
                          initial='http://', required=False)
    upload_type = forms.ChoiceField(widget=forms.RadioSelect(renderer=HorizontalRenderer),
                                    choices=(('file','file'),('url','url')),
                                    initial='file')
    class Meta:
        model = Submission
        fields = ('parity','scale_units','scale_type','scale_lower',
                  'scale_upper','scale_est','scale_err','positional_error',
                  'center_ra','center_dec','radius')
        widgets = {'scale_type': forms.RadioSelect(renderer=HorizontalRenderer),
                   'scale_lower': forms.TextInput(attrs={'size':'5'}),
                   'scale_upper': forms.TextInput(attrs={'size':'5'}),
                   'scale_est': forms.TextInput(attrs={'size':'5'}),
                   'scale_err': forms.TextInput(attrs={'size':'5'}),
                   'positional_error': forms.TextInput(attrs={'size':'5'}),
                   'center_ra': forms.TextInput(attrs={'size':'5'}),
                   'center_dec': forms.TextInput(attrs={'size':'5'}),
                   'radius': forms.TextInput(attrs={'size':'5'}),
                   'parity': forms.RadioSelect(renderer=NoBulletsRenderer),
                  }

    def clean(self):
        upload_type = self.cleaned_data.get('upload_type','')
        if upload_type == 'file':
            if not self.cleaned_data.get('file'):
                raise forms.ValidationError("You must select a file to upload.") 
        elif upload_type == 'url':
            url = self.cleaned_data.get('url','')
            if not (url.startswith('http://') or url.startswith('ftp://')):
                url = 'http://' + url
            if url.startswith('http://http://') or url.startswith('http://ftp://'):
                url = url[7:]
            if len(url) == 0:
                raise forms.ValidationError("You must enter a url to upload.")
            urlvalidator = URLValidator()
            try:
                urlvalidator(url)
            except forms.ValidationError:
                raise forms.ValidationError("You must enter a valid url.")
            self.cleaned_data['url'] = url

        return self.cleaned_data

def upload_file(request):
    if request.method == 'POST':
        form = SubmissionForm(request.POST, request.FILES)
        if form.is_valid():
            sub = form.save(commit=False)
            sub.user = request.user if request.user.is_authenticated() else None
            if form.cleaned_data['upload_type'] == 'file':
                sub.disk_file = handle_upload(file=request.FILES['file'])
                sub.original_filename = request.FILES['file'].name
            elif form.cleaned_data['upload_type'] == 'url':
                sub.url = form.cleaned_data['url']
                sub.disk_file = handle_upload(url=sub.url)
            
            sub.save()
            logmsg('Made Submission' + str(sub))
            return redirect(status, subid=sub.id)
    else:
        form = SubmissionForm()
    return render_to_response('submission/upload.html', {'form': form, 'user': request.user },
        context_instance = RequestContext(request))

def status(req, subid=None):
    logmsg("Submissions: " + ', '.join([str(x) for x in Submission.objects.all()]))

    #if subid is not None:
    #    # strip leading zeros
    #    subid = int(subid.lstrip('0'))
    sub = get_object_or_404(Submission, pk=subid)

    # Would be convenient to have an "is this a single-image submission?" function
    # (This is really "UserImage" status, not Submission status)

    #logmsg("UserImages: " + ', '.join(['%i'%s.id for s in sub.user_images.all()]))

    logmsg("UserImages:")
    for ui in sub.user_images.all():
        logmsg("  %i" % ui.id)
        for j in ui.jobs.all():
            logmsg("    job " + str(j))

    calib = None
    jobs = sub.get_best_jobs()
    logmsg("Best jobs: " + str(jobs))
    #logmsg("Job: " + str(job) + ', ' + job.status)
        
    return render_to_response('submission/status.html',
        {
            'sub': sub,
            'jobs': jobs,
            'calib':calib,
        },
        context_instance = RequestContext(req))
    
def handle_upload(file=None,url=None):
    #logmsg('handle_uploaded_file: req=' + str(req))
    #logmsg('handle_uploaded_file: req.session=' + str(req.session))
    #logmsg('handle_uploaded_file: req.session.user=' + str(req.session.user))
    #logmsg('handle_uploaded_file: req.user=' + str(req.user))

    # get file/url onto disk
    file_hash = DiskFile.get_hash()
    temp_file_path = tempfile.mktemp()
    uploaded_file = open(temp_file_path, 'wb+')

    if file:
        for chunk in file.chunks():
            uploaded_file.write(chunk)
            file_hash.update(chunk)
    elif url:
        f = urllib2.urlopen(url)
        CHUNK_SIZE = 4096
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            uploaded_file.write(chunk)
            file_hash.update(chunk)
    else:
        return None
    uploaded_file.close()

    # get or create DiskFile object
    df,created = DiskFile.objects.get_or_create(file_hash=file_hash.hexdigest(),
                                                defaults={'size':0, 'file_type':''})

    # if the file doesn't already exist, set it's size/type and
    # move file into data directory
    if created:
        DiskFile.make_dirs(file_hash.hexdigest())
        shutil.move(temp_file_path, DiskFile.get_file_path(file_hash.hexdigest()))
        df.set_size_and_file_type()
        df.save()
        
    return df
