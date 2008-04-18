import os
import os.path
import tempfile

from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.template import Context, RequestContext, loader

from astrometry.server.models import *
from astrometry.server.log import log

from astrometry.net.portal.job import Job
from astrometry.net.util.run_command import run_command

def summary(request):
    jobs = QueuedJob.objects.all()
    for j in jobs:
        j.path = Job.s_get_job_dir(j.jobid)
    
    ctxt = {
        'jobqueues': JobQueue.objects.all(),
        'queuedjobs': jobs,
        'workers': Worker.objects.all().order_by('hostname'),
        }
    t = loader.get_template('server/summary.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

def get_input(request):
    job = QueuedJob.objects.get(jobid=request.GET['jobid'])
    path = os.path.join(Job.s_get_job_dir(job.jobid), 'job.axy')
    f = open(path, 'rb')
    res = HttpResponse()
    res['Content-Type'] = 'application/octet-stream'
    res.write(f.read())
    f.close()
    return res

def set_results(request):
    job = QueuedJob.objects.get(jobid=request.GET['jobid'])
    tardata = request.POST['tar']
    log('tardata is %i bytes long' % len(tardata))
    tardata = tardata.decode('base64_codec')
    log('decoded tardata is %i bytes long' % len(tardata))

    # HACK - pipes?
    (f, tarfile) = tempfile.mkstemp('', 'tar-%s-' % job.jobid)
    os.close(f)
    f = open(tarfile, 'wb')
    f.write(tardata)
    f.close()
    log('wrote to tarfile', tarfile)

    outdir = Job.s_get_job_dir(job.jobid)
    cmd = 'cd %s; tar xvf %s' % (outdir, tarfile)
    log('command:', cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        return HttpResponse('tar failed: %i,' % rtn + out + ', ' + err)

    log('set_results: in dir %s, files' % outdir, out.split('\n'))

    return HttpResponse('ok')
