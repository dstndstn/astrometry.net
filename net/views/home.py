from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext

def home(req):
    context = {}
    return render_to_response('home.html', context,
        context_instance = RequestContext(req))

def support(req):
    context = {}
    return render_to_response('support.html', context,
        context_instance = RequestContext(req))
