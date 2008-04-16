from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.template import Context, RequestContext, loader

from astrometry.server.models import *

def summary(request):
    ctxt = {
        'jobqueues': JobQueue.objects.all(),
        'queuedjobs': QueuedJob.objects.all(),
        'workers': Worker.objects.all(),
        }
    t = loader.get_template('server/summary.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))



