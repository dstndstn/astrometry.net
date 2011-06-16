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

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command



def dashboard(request):
    return render_to_response("dashboard.html",
        {
        },
        context_instance = RequestContext(request))


@login_required
def get_api_key(request):
    prof = None
    try:
        prof = request.user.get_profile()
    except UserProfile.DoesNotExist:
        loginfo('Creating new UserProfile for', request.user)
        prof = UserProfile(user=request.user)
        prof.create_api_key()
        prof.save()
        
    context = {
        'apikey':prof.apikey
    }
    return render_to_response("api_key.html",
        context,
        context_instance = RequestContext(request))


