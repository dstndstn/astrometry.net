from django.http import HttpResponse
from django.template import Context, RequestContext, loader
from django import newforms as forms
import an.upload.views as uploadviews
from an.upload.models import UploadedFile
from an.upload.views  import UploadIdField

def view(request):
    uploadform = '/upload/form/'

    # Inline or IFRAME?
    inline = False
    if 'inline' in request.GET:
        inline = True

    if inline:
        progress_meter_html = uploadviews.get_ajax_html()
        progress_meter_js   = uploadviews.get_ajax_javascript()
        progressform = ''
    else:
        progress_meter_html = ''
        progress_meter_js   = ''
        progressform = '/upload/progress_ajax/?upload_id='

    ctxt = {
        'uploadform' : uploadform,
        'progressform' : progressform,
        'progress_meter_html' : progress_meter_html,
        'progress_meter_js' :progress_meter_js,
        'inline_progress' : inline,
        }

    t = loader.get_template('uploaddemo.html')
    c = Context(ctxt)
    return HttpResponse(t.render(c))

class FileForm(forms.Form):
    colour = forms.CharField()
    upload_id = UploadIdField(widget=forms.HiddenInput())

def mainview(request):
    form = FileForm(request.POST)

    if not form.is_valid():
        res = 'form is not valid: '
        for n,e in form._errors.items():
            res += n + ': ' + ';'.join(e)
        return HttpResponse(res)

    uploaded = form.cleaned_data['upload_id']
    colour = form.cleaned_data['colour']
    userfilename = uploaded.userfilename
    filesize = uploaded.filesize
    serverfilename = uploaded.get_filename()
    uploadid = uploaded.uploadid
    
    ctxt = {
        'colour' : colour,
        'uploadid' : uploadid,
        'filesize' : filesize,
        'userfilename' : userfilename,
        'serverfilename' : serverfilename,
        }
    t = loader.get_template('uploaddemomain.html')
    c = Context(ctxt)
    return HttpResponse(t.render(c))

