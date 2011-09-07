from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext
from django.db.models import Count
from astrometry.net.models import *

def home(req):
    context = {
        'images':UserImage.objects.public_only(req.user),
    }
    return render(req, 'home.html', context)

def support(req):
    context = {}
    return render(req, 'support.html', context)

def api_help(req):
    context = {}
    return render(req, 'api_help.html', context)
    
def explore(req):
    recent_images = (UserImage.objects.public_only(req.user)[:12])
    newest_users = User.objects.all().order_by('-date_joined')[:10]
    popular_tags = (Tag.objects.all().annotate(Count('user_images'))
                                     .order_by('-user_images__count')[:35])
    context = {
        'recent_images': recent_images,
        'newest_users': newest_users,
        'popular_tags': popular_tags,
    }
    return render(req, 'explore.html', context)
    
