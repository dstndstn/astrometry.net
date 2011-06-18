import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command

class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        exclude = ('apikey')

def dashboard(request):
    return render_to_response("dashboard.html",
        {
        },
        context_instance = RequestContext(request))

@login_required
def edit_profile(req):
    return render_to_response("dashboard/profile.html",
        {
            'profile':req.user.profile,
        },
        context_instance = RequestContext(req)
    )

@login_required
def save_profile(request):
    profile = request.user.get_profile()
    if request.method == 'POST':
        profile.display_name = request.POST['display_name']
        profile.save()
    return redirect('astrometry.net.views.user.dashboard_profile')

@login_required
def dashboard_profile(request):
    profile = None
    try:
        profile = request.user.get_profile()
        form = ProfileForm(instance=profile)
    except UserProfile.DoesNotExist:
        loginfo('Creating new UserProfile for', request.user)
        profile = UserProfile(user=request.user)
        profile.create_api_key()
        profile.save()
        
    context = {
        'profile_form':form,
        'profile':profile,
    }
    return render_to_response("dashboard/profile.html",
        context,
        context_instance = RequestContext(request))

@login_required
def dashboard_submissions(req):
    context = {
        'user_submissions':req.user.submissions.all
    }
    return render_to_response("dashboard/submissions.html",
        context,
        context_instance = RequestContext(req))

def public_profile(req):
    pass
