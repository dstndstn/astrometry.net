from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

def remove_userimagetag(req, user_image_id, tag_id):
    tag = get_object_or_404(TaggedUserImage, user_image=user_image_id, tag=tag_id)
    tag.delete()
    redirect_url = reverse('astrometry.net.views.home.home')
    if 'next' in req.GET:
        redirect_url = req.GET['next']
    return HttpResponseRedirect(redirect_url)
