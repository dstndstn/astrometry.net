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
from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.forms.models import inlineformset_factory
from django.http import HttpResponseRedirect

from astrometry.net.util import HorizontalRenderer, NoBulletsRenderer
from astrometry.util.run_command import run_command
from urlparse import urlparse

def index(req, user_id):
    submitter = None
    if user_id == None:
        submissions = Submission.objects.all()
    else:
        submitter = get_object_or_404(User, pk=user_id)
    
    context = {'submitter':submitter}
    return render_to_response('submission/by_user.html', context,
        context_instance = RequestContext(req))

class SubmissionForm(forms.ModelForm):
    SCALE_PRESET_SETTINGS = {'1':(0.1,180),
                             '2':(1,180),
                             '3':(10,180),
                             '4':(0.05,0.015)}

    file  = forms.FileField(required=False,
                            widget=forms.FileInput(attrs={'size':'80'}))
    url = forms.CharField(widget=forms.TextInput(attrs={'size':'80'}),
                          initial='http://', required=False)
    upload_type = forms.ChoiceField(widget=forms.RadioSelect(renderer=HorizontalRenderer),
                                    choices=(('file','file'),('url','url')),
                                    initial='file')
    scale_preset = forms.ChoiceField(widget=forms.RadioSelect(renderer=NoBulletsRenderer),
                                    choices=(('1','default (0.1 to 180 degrees)'),
                                             ('2','wide field (1 to 180 degrees)'),
                                             ('3','very wide field (10 to 180 degrees)'),
                                             ('4','tiny (3 to 9 arcmin)'),
                                             ('5','custom')),
                                    initial='1')

    album = forms.ChoiceField(choices=(), required=False)
    new_album_title = forms.CharField(widget=forms.TextInput(attrs={'size':'40'}),
                                      max_length=64,
                                      required=False)

    class Meta:
        model = Submission
        fields = (
            'publicly_visible',
            'allow_commercial_use', 'allow_modifications',
            'parity','scale_units','scale_type','scale_lower',
            'scale_upper','scale_est','scale_err','positional_error',
            'center_ra','center_dec','radius','downsample_factor','source_type')
        widgets = {
            'scale_type': forms.RadioSelect(renderer=HorizontalRenderer),
            'scale_lower': forms.TextInput(attrs={'size':'5'}),
            'scale_upper': forms.TextInput(attrs={'size':'5'}),
            'scale_est': forms.TextInput(attrs={'size':'5'}),
            'scale_err': forms.TextInput(attrs={'size':'5'}),
            'positional_error': forms.TextInput(attrs={'size':'5'}),
            'center_ra': forms.TextInput(attrs={'size':'5'}),
            'center_dec': forms.TextInput(attrs={'size':'5'}),
            'radius': forms.TextInput(attrs={'size':'5'}),
            'downsample_factor': forms.TextInput(attrs={'size':'5'}),
            'parity': forms.RadioSelect(renderer=NoBulletsRenderer),
            'source_type': forms.RadioSelect(renderer=NoBulletsRenderer),
            'publicly_visible': forms.RadioSelect(renderer=NoBulletsRenderer),
            'allow_commercial_use':forms.RadioSelect(renderer=NoBulletsRenderer),
            'allow_modifications':forms.RadioSelect(renderer=NoBulletsRenderer),
        }

    def clean(self):
        number_message = "Enter a number."

        scale_preset = self.cleaned_data.get('scale_preset','')
        if scale_preset == '5':
            # custom preset error handling
            scale_type = self.cleaned_data.get('scale_type','')
            if scale_type == 'ul':
                scale_lower = self.cleaned_data.get('scale_lower')
                scale_upper = self.cleaned_data.get('scale_upper')
                if not scale_lower:
                    self._errors['scale_lower'] = self.error_class([number_message])
                if not scale_upper:
                    self._errors['scale_upper'] = self.error_class([number_message])
            elif scale_type == 'ev':
                scale_est = self.cleaned_data.get('scale_est')
                scale_err = self.cleaned_data.get('scale_err')
                if not scale_est:
                    self._errors['scale_est'] = self.error_class([number_message])
                if not scale_err:
                    self._errors['scale_err'] = self.error_class([number_message])
        else:
            # if scale isn't custom, use preset settings
            self.cleaned_data['scale_type'] = 'ul'
            self.cleaned_data['scale_units'] = 'degwidth'
            self.cleaned_data['scale_lower'] = self.SCALE_PRESET_SETTINGS[scale_preset][0]
            self.cleaned_data['scale_upper'] = self.SCALE_PRESET_SETTINGS[scale_preset][1]

        center_ra = self.cleaned_data.get('center_ra')
        center_dec = self.cleaned_data.get('center_dec')
        radius = self.cleaned_data.get('radius')
        if center_ra or center_dec or radius:
            if not center_ra:
                self._errors['center_ra'] = self.error_class([number_message])
            if not center_dec:
                self._errors['center_dec'] = self.error_class([number_message])
            if not radius:
                self._errors['radius'] = self.error_class([number_message])

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
        
    def __init__(self, user, *args, **kwargs):
        super(SubmissionForm, self).__init__(*args, **kwargs)
        #if user.is_authenticated():
        #    self.fields['album'].queryset = user.albums
        self.fields['album'].choices = [('', 'none')]
        if user.is_authenticated():
            for album in Album.objects.filter(user=user).all():
                self.fields['album'].choices += [(album.id, album.title)]
            self.fields['album'].choices += [('new', 'create new album...')]
        self.fields['album'].initial = ''


def upload_file(request):
    default_license = License.get_default()
    if request.user.is_authenticated():
        default_license = request.user.get_profile().default_license 

    if request.method == 'POST':
        form = SubmissionForm(request.user, request.POST, request.FILES)
        if form.is_valid():
            sub = form.save(commit=False)
            
            if request.user.is_authenticated():
                if form.cleaned_data['album'] == '':
                    # don't create an album
                    pass
                elif form.cleaned_data['album'] == 'new':
                    # create a new album
                    title = form.cleaned_data['new_album_title']
                    if title:
                        album,created = Album.objects.get_or_create(user=request.user, title=title)
                        sub.album = album
                else:
                    try:
                        sub.album = Album.objects.get(pk=int(form.cleaned_data['album']))
                    except Exception as e:
                        print e

            if not request.user.is_authenticated():
                sub.publicly_visible = 'y'
            if form.cleaned_data['allow_commercial_use'] == 'd':
                sub.allow_commercial_use = default_license.allow_commercial_use
            if form.cleaned_data['allow_modifications'] == 'd':
                sub.allow_modifications = default_license.allow_modifications

            sub.user = request.user if request.user.is_authenticated() else User.objects.get(username=ANONYMOUS_USERNAME)
            if form.cleaned_data['upload_type'] == 'file':
                sub.disk_file, sub.original_filename = handle_upload(file=request.FILES['file'])
            elif form.cleaned_data['upload_type'] == 'url':
                sub.url = form.cleaned_data['url']
                sub.disk_file, sub.original_filename = handle_upload(url=sub.url)
            
            sub.save()
            logmsg('Made Submission' + str(sub))
            return redirect(status, subid=sub.id)
    else:
        form = SubmissionForm(request.user)
    return render_to_response('submission/upload.html',
        {
            'form': form,
            'user': request.user,
            'default_license': default_license,
        },
        context_instance = RequestContext(request))

def job_log_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_log_file())
    res = HttpResponse(f)
    res['Content-type'] = 'text/plain'
    return res

def job_log_file2(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_log_file2())
    res = HttpResponse(f)
    res['Content-type'] = 'text/plain'
    return res

def status(req, subid=None):
    logmsg("Submissions: " + ', '.join([str(x) for x in Submission.objects.all()]))
    sub = get_object_or_404(Submission, pk=subid)

    # Would be convenient to have an "is this a single-image submission?" function
    # (This is really "UserImage" status, not Submission status)

    #logmsg("UserImages: " + ', '.join(['%i'%s.id for s in sub.user_images.all()]))

    # might want to make this a field in Submission
    finished = (len(sub.user_images.all()) > 0)
    if finished == False and sub.processing_finished != None:
        finished = True
    logmsg("UserImages:")
    for ui in sub.user_images.all():
        logmsg("  %i" % ui.id)
        if len(ui.jobs.all()) == 0:
            finished = False
        else:
            for j in ui.jobs.all():
                logmsg("    job " + str(j))
                if j.end_time is None:
                    finished = False
     

    return render_to_response('submission/status.html',
        {
            'sub': sub,
            'anonymous_username':ANONYMOUS_USERNAME,
            'finished': finished,
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
    original_filename = ''

    if file:
        for chunk in file.chunks():
            uploaded_file.write(chunk)
            file_hash.update(chunk)
        original_filename = file.name
    elif url:
        f = urllib2.urlopen(url)
        CHUNK_SIZE = 4096
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            uploaded_file.write(chunk)
            file_hash.update(chunk)

        p = urlparse(url)
        p = p.path
        if p:
            s = p.split('/')
            original_filename = s[-1]
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
        
    return df, original_filename
