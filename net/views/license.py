from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

class LicenseForm(forms.ModelForm):
    class Meta:
        model = Licensable


class PartialLicenseForm(forms.ModelForm):
    class Meta:
        model = Licensable
        exclude = ('license_name','license_uri')

@login_required
def edit(req, licensable_type=None, licensable_id=None):
    if req.method == 'POST':
        try:
            types = {
                'UserImage':UserImage,
                'License':License
            }
            licensee = get_object_or_404(types[licensable_type], pk=licensable_id)
            default_license = License.get_default()

            allow_commercial = req.POST.get('allow_commercial_use',default_license.allow_commercial_use)
            allow_mod = req.POST.get('allow_modifications',default_license.allow_modifications)

            licensee.allow_commercial_use = allow_commercial
            licensee.allow_modifications = allow_mod
            licensee.save()
            redirect_url = req.POST.get('next','/')
        except:
            print 'failed'
            redirect_url = ('/')
            pass

        return HttpResponseRedirect(redirect_url)
    else:
        # show a generic license form?
        pass
