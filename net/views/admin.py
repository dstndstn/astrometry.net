from datetime import datetime
from django.shortcuts import get_object_or_404

from astrometry.net.models import ProcessSubmissions
from astrometry.net.log import logmsg

def index(req):
    ps = ProcessSubmissions.objects.all().order_by('-watchdog')
    logmsg('ProcessSubmissions:', ps)
    return render(req, 'admin.html', {'procsubs': ps,})

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
    return render(req, 'procsub.html', 'procsub': ps, 'now': now,})
