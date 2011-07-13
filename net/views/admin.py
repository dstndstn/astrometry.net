
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from astrometry.net.tmpfile import *

def index(req):
    ps = ProcessSubmissions.objects.all().order_by('-watchdog')
    logmsg('ProcessSubmissions:', ps)
    return render_to_response('admin.html',
                              { 'procsubs':ps,
                                },
                              context_instance = RequestContext(req))

def procsub(req, psid=None):
    ps = get_object_or_404(ProcessSubmissions, pk=psid)
    logmsg('ProcessSubmission:', ps)
    logmsg('jobs:', ps.jobs.all())
    for j in ps.jobs.all():
        logmsg('  ', j)
        logmsg('  ', j.job)
        logmsg('  ', j.job.user_image)
        logmsg('  ', j.job.user_image.submission)
    now = datetime.now()
    now = now.replace(microsecond=0)
    now = now.isoformat()
    return render_to_response('procsub.html',
                              { 'procsub':ps,
                                'now':now,
                                },
                              context_instance = RequestContext(req))
