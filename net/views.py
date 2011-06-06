from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response
from django.template import Context, RequestContext, loader

def dashboard(request):
    return render_to_response("dashboard.html",
        {
		},
		context_instance = RequestContext(request))
