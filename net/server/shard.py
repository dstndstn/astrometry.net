import os
import os.path
import socket

from django.http import HttpResponse

from astrometry.net.server.log import log
from astrometry.net.server.models import QueuedJob, Worker, JobQueue, Index
from astrometry.net.portal.job import Job, Calibration
from astrometry.net.portal.wcs import TanWCS
from astrometry.util.run_command import run_command
from astrometry.util.file import *

import astrometry.net.settings as settings

def tempdir_for_jobid(jobid):
    # FIXME
    return '/tmp/backend-results-%s' % jobid


def solve(request):
    log('shard.solve')

    if not 'axy' in request.POST:
        return HttpResponse('no axy')
    if not 'jobid' in request.POST:
        return HttpResponse('no jobid')

    jobid = request.POST['jobid']
    axy = request.POST['axy']
    #axy = axy.decode('base64_codec')

    tmpdir = tempdir_for_jobid(jobid)
    os.mkdir(tmpdir)
    axyfn = os.path.join(tmpdir, 'input.axy')
    write_file(axy, axyfn)

    cancelfile = os.path.join(tmpdir, 'cancel')
    tarfile = os.path.join(tmpdir, 'results.tar')

    backendcfg = settings.BACKEND_CONFIG % socket.gethostname()

    cmd = ('cd %(tempdir)s; '
           '%(backend)s -c %(backendconf)s -C %(cancel)s -v %(axy)s > %(logfile)s 2>&1; '
           'tar cf %(tarfile)s *' %
           dict(tempdir=tmpdir, backend='backend',
                backendconf=backendcfg,
                cancel=cancelfile,
                axy=axyfn,
                logfile='backend.log',
                tarfile=tarfile))

    print 'Running command', cmd
    
    (rtn, out, err) = run_command(cmd)
    if rtn:
        print 'backend failed: rtn val %i' % rtn, ', out', out, ', err', err

    res = HttpResponse()
    res['Content-type'] = 'application/x-tar'
    res.write(read_file(tarfile))
    return res

def cancel(request):
    jobid = request.POST.get('jobid')
    if not jobid:
        return HttpResponseBadRequest('no jobid')

    tmpdir = tempdir_for_jobid(jobid)
    cancelfile = os.path.join(tmpdir, 'cancel')
    log('shard.cancel jobid', jobid)
    # FIXME - security hole.
    write_file(' ', cancelfile)
    return HttpResponse('ok')
