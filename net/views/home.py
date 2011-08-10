from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext
from astrometry.net.models import *

def home(req):
    context = {
        'images':UserImage.objects.all().order_by('-submission__submitted_on'),
    }
    return render(req, 'home.html', context)

def support(req):
    context = {}
    return render(req, 'support.html', context)

def api_help(req):
    context = {}
    return render(req, 'api_help.html', context)
