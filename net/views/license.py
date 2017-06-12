from __future__ import print_function
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from django import forms
from django.http import HttpResponseRedirect

class LicenseForm(forms.ModelForm):
    class Meta:
        model = License
        exclude = ('license_uri','license_name')
        widgets = {
            'allow_commercial_use':forms.RadioSelect(template='radio-nobullets.html'), #renderer=NoBulletsRenderer),
            'allow_modifications':forms.RadioSelect(template='radio-nobullets.html'), #renderer=NoBulletsRenderer),
        }

@login_required
def edit(req, license_id):
    if req.method == 'POST':
        try:
            license = get_object_or_404(License, pk=license_id)
            default_license = License.get_default()

            allow_commercial = req.POST.get('allow_commercial_use',default_license.allow_commercial_use)
            allow_mod = req.POST.get('allow_modifications',default_license.allow_modifications)

            license.allow_commercial_use = allow_commercial
            license.allow_modifications = allow_mod
            license.save()
            redirect_url = req.POST.get('next','/')
        except:
            print('failed')
            redirect_url = ('/')
            pass

        return HttpResponseRedirect(redirect_url)
    else:
        # show a generic license form?
        pass
