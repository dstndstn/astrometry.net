import re
import os

from django import newforms as forms
from django.db import models
from django.http import HttpResponse, HttpResponseRedirect
from django.newforms import widgets, ValidationError, form_for_model
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response
from django.contrib.auth.decorators import login_required

import astrometry

import astrometry.net.upload.views as uploadviews

from astrometry.net.upload.models import UploadedFile
from astrometry.net.upload.views  import UploadIdField

from astrometry.net import settings

from astrometry.net.portal.job import Submission, Job
from astrometry.net.portal.log import log
from astrometry.net.portal.views import printvals, get_status_url

ftpurl_re = re.compile(
    r'^ftp://'
    r'(?:(?:[A-Z0-9-]+\.)+[A-Z]{2,6}|' #domain...
    r'localhost|' #localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?' # optional port
    r'(?:/?|/\S+)$', re.IGNORECASE)

class FtpOrHttpURLField(forms.URLField):
    def clean(self, value):
        #log('FtpOrHttpURLField.clean(', value, ')')
        try:
            val = super(FtpOrHttpURLField, self).clean(value)
            return val
        except ValidationError:
            pass
        if ftpurl_re.match(value):
            return value
        raise ValidationError(self.error_messages['invalid'])

class ForgivingURLField(FtpOrHttpURLField):
    def clean(self, value):
        #log('ForgivingURLField.clean(', value, ')')
        if value is not None and \
               (value.startswith('http://http://') or
                value.startswith('http://ftp://')):
            value = value[7:]
        return super(ForgivingURLField, self).clean(value)

class SimpleURLForm(forms.Form):
    url = ForgivingURLField(initial='http://',
                            widget=forms.TextInput(attrs={'size':'50'}))

class SimpleFancyFileForm(forms.Form):
    upload_id = UploadIdField(widget=forms.HiddenInput())

class FullForm(forms.Form):
    
    datasrc = forms.ChoiceField(choices=Submission.datasrc_CHOICES,
                                initial='url',
                                widget=forms.RadioSelect(
        attrs={'id':'datasrc',
               'onclick':'datasourceChanged()'}))

    filetype = forms.ChoiceField(choices=Submission.filetype_CHOICES,
                                 initial='image',
                                 widget=forms.Select(
        attrs={'onchange':'filetypeChanged()',
               'onkeyup':'filetypeChanged()'}))

    upload_id = UploadIdField(widget=forms.HiddenInput(),
                              required=False)

    scaleunits = forms.ChoiceField(choices=Submission.scaleunits_CHOICES,
                                   initial=Submission.scaleunits_default,
                                   widget=forms.Select(
        attrs={'onchange':'unitsChanged()',
               'onkeyup':'unitsChanged()'}))
    scaletype = forms.ChoiceField(choices=Submission.scaletype_CHOICES,
                                  widget=forms.RadioSelect(
        attrs={'id':'scaletype',
               'onclick':'scalechanged()'}),
                                  initial='ul')
    parity = forms.ChoiceField(choices=Submission.parity_CHOICES,
                               initial=2)

    description = forms.CharField(widget=forms.Textarea(
        attrs={'rows':2,}
        ), required=False)

    scalelower = forms.DecimalField(widget=forms.TextInput(
        attrs={'onfocus':'setFsUl()',
               'onkeyup':'scalechanged()',
               'size':'5',
               }), initial=0.1, required=False, min_value=0)
    scaleupper = forms.DecimalField(widget=forms.TextInput(
        attrs={'onfocus':'setFsUl()',
               'onkeyup':'scalechanged()',
               'size':'5',
               }), initial=180, required=False, min_value=0)
    scaleest = forms.DecimalField(widget=forms.TextInput(
        attrs={'onfocus':'setFsEv()',
               'onkeyup':'scalechanged()',
               'size':'5',
               }), required=False, min_value=0)
    scaleerr = forms.DecimalField(widget=forms.TextInput(
        attrs={'onfocus':'setFsEv()',
               'onkeyup':'scalechanged()',
               'size':'5',
               }), required=False, min_value=0, max_value=99)

    url = ForgivingURLField(initial='http://',
                            widget=forms.TextInput(
        attrs={'size':'50',
               'onfocus':'setDatasourceUrl()',
               'onclick':'setDatasourceUrl()',
               }),
                            required=False)
    xcol = forms.CharField(initial='X',
                           required=False,
                           widget=forms.TextInput(attrs={'size':'10'}))
    ycol = forms.CharField(initial='Y',
                           required=False,
                           widget=forms.TextInput(attrs={'size':'10'}))

    tweak = forms.BooleanField(initial=True, required=False)

    tweakorder = forms.IntegerField(initial=2, min_value=2,
                                    max_value=10,
                                    widget=forms.TextInput(attrs={'size':'5'}))

    # How to clean an individual field:
    #def clean_scaletype(self):
    #   val = self.getclean('scaletype')
    #   return val

    def getclean(self, name):
        if not hasattr(self, 'cleaned_data'):
            return None
        if name in self.cleaned_data:
            return self.cleaned_data[name]
        return None

    def geterror(self, name):
        if name in self._errors:
            return self._errors[name]
        return None

    def reclean(self, name):
        field = self.fields[name].field
        value = field.widget.value_from_datadict(self.data, self.files, self.add_prefix(name))
        try:
            value = field.clean(value)
            self.cleaned_data[name] = value
            if hasattr(self, 'clean_%s' % name):
                value = getattr(self, 'clean_%s' % name)()
                self.cleaned_data[name] = value
        except ValidationError, e:
            self._errors[name] = e.messages
            if name in self.cleaned_data:
                del self.cleaned_data[name]             

    def clean(self):
        """Take a shower"""
        src = self.getclean('datasrc')
        if src == 'url':
            if self.geterror('file'):
                del self._errors['file']
            if self.geterror('upload_id'):
                del self._errors['upload_id']
            url = self.getclean('url')
            log('url is ' + str(url))
            if not self.geterror('url'):
                if not (url and len(url)):
                    self._errors['url'] = ['URL is required']
                elif not (url.startswith('http://') or url.startswith('ftp://')):
                    self._errors['url'] = ['Only http and ftp URLs are supported']

        elif src == 'file':
            if self.geterror('url'):
                del self._errors['url']
            fil = self.getclean('file')
            uploadid = self.getclean('upload_id')
            # FIXME - umm... little more checking here :)
            log('file is ' + str(fil) + ', upload id is ' + str(uploadid))
            if not (fil or uploadid):
                self._errors['file'] = ['You must upload a file']

        filetype = self.getclean('filetype')
        if filetype == 'fits':
            xcol = self.getclean('xcol')
            ycol = self.getclean('ycol')
            fitscol = re.compile('^[\w_-]+$')
            if not self.geterror('xcol'):
                if not xcol or len(xcol) == 0:
                    self._errors['xcol'] = ['You must specify the X column name.']
                elif not fitscol.match(xcol):
                    self._errors['xcol'] = ['X column name must be alphanumeric.']
            if not self.geterror('ycol'):
                if not ycol or len(ycol) == 0:
                    self._errors['ycol'] = ['You must specify the Y column name.']
                elif not fitscol.match(ycol):
                    self._errors['ycol'] = ['Y column name must be alphanumeric.']

        if self.getclean('tweak'):
            order = self.getclean('tweakorder')
            print 'tweak order', order
            if not order:
                self._errors['tweakorder'] = ['Tweak order is required if tweak is enabled.']

        log('form clean(): scaletype is %s' % self.getclean('scaletype'))
        scaletype = self.getclean('scaletype')
        if scaletype == 'ul':
            if not self.geterror('scalelower') and not self.geterror('scaleupper'):
                lower = self.getclean('scalelower')
                upper = self.getclean('scaleupper')
                if lower is None:
                    self._errors['scalelower'] = ['You must enter a lower bound.']
                if upper is None:
                    self._errors['scaleupper'] = ['You must enter an upper bound.']
                if (lower > upper):
                    self._errors['scalelower'] = ['Lower bound must be less than upper bound.']
        elif scaletype == 'ev':
            if not self.geterror('scaleest') and not self.geterror('scaleerr'):
                est = self.getclean('scaleest')
                err = self.getclean('scaleerr')
                if est is None:
                    self._errors['scaleest'] = ['You must enter a scale estimate.']
                if err is None:
                    self._errors['scaleerr'] = ['You must enter a scale error bound.']

        # If tweak is off, ignore errors in tweakorder.
        if self.getclean('tweak') == False:
            if self.geterror('tweakorder'):
                del self._errors['tweakorder']
                self.cleaned_data['tweakorder'] = 0

        return self.cleaned_data

def submit_submission(request, submission):
    log('submit_submission(): Submission is: ' + str(submission))
    submission.set_submittime_now()
    submission.set_status('Queued')
    submission.save()
    request.session['jobid'] = submission.get_id()
    Job.submit_job_or_submission(submission)

@login_required
def newurl(request):
    urlerr = None
    if len(request.POST):
        form = SimpleURLForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['url']
            submission = Submission(user = request.user,
                            filetype = 'image',
                            datasrc = 'url',
                            url = url)
            submission.save()
            submit_submission(request, submission)
            return HttpResponseRedirect(get_status_url(submission.subid))
        else:
            urlerr = form['url'].erroros[0]
    else:
        if 'jobid' in request.session:
            del request.session['jobid']
        form = SimpleURLForm()

    return render_to_response(
        'portal/newjoburl.html',
        {
        'form' : form,
        'urlerr' : urlerr,
        'actionurl': reverse(newurl),
        },
        context_instance = RequestContext(request))
        
    #t = loader.get_template('portal/newjoburl.html')
    #c = RequestContext(request, {
    #    'form' : form,
    #    'urlerr' : urlerr,
    #    })
    #return HttpResponse(t.render(c))

def uploadformurl():
    return (reverse(astrometry.net.upload.views.uploadform)
            + '?onload=parent.uploadframeloaded()'
            + '&onload2=parent.uploadFinished()'
            )

@login_required
def newfile(request):
    if len(request.POST):
        form = SimpleFancyFileForm(request.POST)
        if form.is_valid():
            submission = Submission(user = request.user,
                            filetype = 'image',
                            datasrc = 'file',
                            uploaded = form.cleaned_data['upload_id'],
                            )
            submission.save()
            submit_submission(request, submission)
            return HttpResponseRedirect(get_status_url(submission.subid))
        else:
            log('form not valid.')
    else:
        if 'jobid' in request.session:
            del request.session['jobid']
        form = SimpleFancyFileForm()
        
    t = loader.get_template('portal/newjobfile.html')
    c = RequestContext(request, {
        'form' : form,
        'uploadform' : uploadformurl(),
        'progressform' : reverse(astrometry.net.upload.views.progress_ajax) + '?upload_id='
        })
    return HttpResponse(t.render(c))

# Note, if there are *ANY* errors in the form, it will have no
# 'cleaned_data' array.

@login_required
def newlong(request):
    if request.POST:
        form = FullForm(request.POST, request.FILES)
    else:
        form = FullForm()

    if form.is_valid():
        uploaded = form.getclean('upload_id')
        url = form.getclean('url')
        if form.getclean('datasrc') == 'url':
            uploaded = None
        elif form.getclean('datasrc') == 'file':
            url = None

        submission = Submission(
            user = request.user,
            description = form.getclean('description'),
            datasrc = form.getclean('datasrc'),
            filetype = form.getclean('filetype'),
            uploaded = uploaded,
            url = url,
            xcol = form.getclean('xcol'),
            ycol = form.getclean('ycol'),
            parity = form.getclean('parity'),
            scaleunits = form.getclean('scaleunits'),
            scaletype = form.getclean('scaletype'),
            scalelower = form.getclean('scalelower'),
            scaleupper = form.getclean('scaleupper'),
            scaleest = form.getclean('scaleest'),
            scaleerr = form.getclean('scaleerr'),
            tweak = form.getclean('tweak'),
            tweakorder = form.getclean('tweakorder'),
            )
        submission.save()
        submit_submission(request, submission)
        return HttpResponseRedirect(get_status_url(submission.subid))

    if 'jobid' in request.session:
        del request.session['jobid']

    log('Errors:')
    if form._errors:
        for k,v in form._errors.items():
            log('  %s = %s' % (str(k), str(v)))

    # Ugh, special rendering for radio checkboxes....
    scaletype = form['scaletype'].field
    widg = scaletype.widget
    attrs = widg.attrs
    if request.POST:
        val = widg.value_from_datadict(request.POST, None, 'scaletype')
    else:
        val = form.getclean('scaletype') or scaletype.initial
    render = widg.get_renderer('scaletype', val, attrs)
    r0txt = render[0].tag()
    r1txt = render[1].tag()

    datasrc = form['datasrc'].field
    widg = datasrc.widget
    attrs = widg.attrs
    if request.POST:
        val = widg.value_from_datadict(request.POST, None, 'datasrc')
    else:
        val = form.getclean('datasrc') or datasrc.initial
    render = widg.get_renderer('datasrc', val, attrs)
    ds0 = render[0].tag()
    ds1 = render[1].tag()

    inline = False

    if inline:
        progress_meter_html = upload.views.get_ajax_html()
        progress_meter_js   = upload.views.get_ajax_javascript()
        progressform = ''
    else:
        progress_meter_html = ''
        progress_meter_js   = ''
        progressform = reverse(uploadviews.progress_ajax) + '?upload_id='

    form['url'].field.widget.attrs['id'] = 'id_img_url'
    imgurlinput = str(form['url'])
    form['url'].field.widget.attrs['id'] = 'id_fitsimg_url'
    fitsurlinput = str(form['url'])
    form['url'].field.widget.attrs['id'] = 'id_url'

    ctxt = {
        'form' : form,
        'imgurlinput': imgurlinput,
        'fitsurlinput': fitsurlinput,
        'uploadform' : uploadformurl(),
        'progressform' : progressform,
        'myurl' : reverse(astrometry.net.portal.newjob.newlong),
        'scale_ul' : r0txt,
        'scale_ee' : r1txt,
        'datasrc_url' : ds0,
        'datasrc_file' : ds1,
        'progress_meter_html' : progress_meter_html,
        'progress_meter_js' :progress_meter_js,
        'inline_progress' : inline,
        }

    errfields = [ 'scalelower', 'scaleupper', 'scaleest', 'scaleerr',
                  'tweakorder', 'xcol', 'ycol', 'scaletype', 'datasrc',
                  'url' ]
    for f in errfields:
        ctxt[f + '_err'] = len(form[f].errors) and form[f].errors[0] or None

    if request.POST:
        if request.FILES:
            f1 = request.FILES['file']
            print 'file is', repr(f1)
        else:
            print 'no file uploaded'

    t = loader.get_template('portal/newjoblong.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

