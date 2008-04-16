import urllib
import tempfile
import os

from django import newforms as forms
from django.http import HttpResponse, HttpResponseRedirect
from django.newforms import widgets, ValidationError, form_for_model
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse

from astrometry.net.util.run_command import run_command
from astrometry.net.util.shell import shell_escape, shell_escape_inside_quotes
from astrometry.net.portal.log import log

class HoggThinksForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea())

def form(request):
    imgurl = None
    form = None
    if len(request.GET):
        form = HoggThinksForm(request.GET)
        if form.is_valid():
            imgurl = reverse(image) + '?' + urllib.urlencode(form.cleaned_data) #{'text':form.cleaned_data['text']})

    if not form:
        form = HoggThinksForm()
    t = loader.get_template('portal/hoggthinks.html')
    c = RequestContext(request, {
        'form' : form,
        'imgurl' : imgurl,
        })
    return HttpResponse(t.render(c))

def image(request):
    if not len(request.GET):
        return HttpResponse('no text')
    form = HoggThinksForm(request.GET)
    if not form.is_valid():
        return HttpResponse('invalid form')
    (f, tmpfile) = tempfile.mkstemp('.jpg', 'hoggthinks')
    os.close(f)
    img = gmaps_config.tcdir + 'an/portal/hogg-thinks.jpg'
    cmd = ('add-text -o %s -j %s -x 425 -y 170 -W 360 -H 190 -c black -t "%s"'
           % (tmpfile, img, shell_escape_inside_quotes(form.cleaned_data['text'])))
    (rtn, stdout, stderr) = run_command(cmd)
    res = HttpResponse()
    #log('txt: "' + form.cleaned_data['text'] + '"')
    #log('escaped: "' + shell_escape(form.cleaned_data['text']) + '"')
    if rtn:
        res.write('Command failed: cmd=%s, stdout=%s, stderr=%s' % (cmd, stdout, stderr))
        return res
    res['Content-Type'] = 'image/jpeg'
    res['Content-Disposition'] = 'inline'
    f = open(tmpfile)
    res.write(f.read())
    f.close()
    os.unlink(tmpfile)
    return res


