import django.contrib.auth as auth
from django.contrib.auth.decorators import login_required

from django.db import models
from django.http import HttpResponse, HttpResponseRedirect
from django.newforms import widgets, ValidationError, form_for_model
from django.template import Context, RequestContext, loader

from an.testbed.models import *

@login_required
def oldjobs(request):
    ojs = OldJob.objects.all()

    ctxt = {
        'oldjobs' : ojs,
        }
    t = loader.get_template('testbed/oldjobs.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

