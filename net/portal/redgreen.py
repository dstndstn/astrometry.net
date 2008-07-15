from django import newforms as forms
from django.http import HttpResponse, HttpResponseRedirect
from django.template import Context, RequestContext, loader

from astrometry.net.portal.views import getfile

culurs=[(c,c) for c in ['red','green', 'blue', 'white',
                        'black', 'cyan', 'magenta', 'yellow',
                        'brightred', 'skyblue', 'orange']]
markurs=[(m,m) for m in [ 'circle', 'crosshair', 'square', 'diamond', 'X', 'Xcrosshair' ]]

class RedGreenForm(forms.Form):
    red   = forms.ChoiceField(choices=culurs, initial='red')
    green = forms.ChoiceField(choices=culurs, initial='green')
    rmarker = forms.ChoiceField(choices=markurs, initial='circle')
    gmarker = forms.ChoiceField(choices=markurs, initial='circle')
    redhex   = forms.CharField(required=False)#, attrs={'size':6})
    greenhex = forms.CharField(required=False)#, attrs={'size':6})

def redgreen(request):
    if request.GET:
        form = RedGreenForm(request.GET)
    else:
        form = RedGreenForm()
    if form.is_valid():
        red = form.cleaned_data['redhex'] or form.cleaned_data['red']
        green = form.cleaned_data['greenhex'] or form.cleaned_data['green']
        # HACK
        return HttpResponseRedirect(reverse(getfile, args=['test-200802-02380922', 'redgreen']) + '?red=%s&green=%s&rmarker=%s&gmarker=%s' %
                                    (red, green,
                                     form.cleaned_data['rmarker'], form.cleaned_data['gmarker']))
                                     
    ctxt = {
        'form' : form,
        }
    t = loader.get_template('portal/redgreen.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

