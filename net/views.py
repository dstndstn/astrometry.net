
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from log import *

def dashboard(request):
    return render_to_response("dashboard.html",
        {
		},
		context_instance = RequestContext(request))


@login_required
def get_api_key(request):
	try:
		prof = request.user.get_profile()
	except UserProfile.DoesNotExist:
		loginfo('Creating new UserProfile for', request.user)
		prof = UserProfile(user=request.user)
		prof.create_api_key()
		prof.save()
		
	return HttpResponse(str(prof))

