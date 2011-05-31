from django.http import HttpResponse, HttpResponseRedirect
from astrometry.net1.portal.views import get_status_url
from astrometry.net1.portal.log import log

def jobstatus_old(request):
    if not request.GET:
        return HttpResponse('no GET')
    if not 'jobid' in request.GET:
        return HttpResponse('no jobid')
    jobid = request.GET['jobid']
    log('jobstatus_old: jobid ' + jobid + ' redirecting to ' +
        get_status_url(jobid))
    return HttpResponseRedirect(get_status_url(jobid))

