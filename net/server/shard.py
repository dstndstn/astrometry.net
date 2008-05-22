import os
import os.path
import socket

from django.http import HttpResponse

from astrometry.net.server.log import log
from astrometry.util.run_command import run_command
from astrometry.util.file import *

import astrometry.net.settings as settings

def tempdir_for_jobid(jobid):
    # FIXME
    return '/tmp/backend-results-%s' % jobid


def solve(request):
    log('shard.solve')

    jobid = request.POST.get('jobid')
    if not jobid:
        return HttpResponse('no jobid')
    axy = request.POST.get('axy')
    if not axy:
        return HttpResponse('no axy')
    # FIXME
    axy = axy.decode('base64_codec')

    tmpdir = tempdir_for_jobid(jobid)
    print 'tmpdir is',tmpdir
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    print 'made tmpdir.'
    axyfn = os.path.join(tmpdir, 'input.axy')
    print 'axyfn is', axyfn
    try:
        write_file(axy, axyfn)
    except Exception, e:
        print 'something failed',e
    print 'wrote axy'

    cancelfile = os.path.join(tmpdir, 'cancel')
    tarfile = os.path.join(tmpdir, 'results.tar')

    backendcfg = settings.BACKEND_CONFIG % socket.gethostname()
    #backendcfg = settings.BACKEND_CONFIG

    print 'backendcfg', backendcfg

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
    jobid = request.GET.get('jobid')
    if not jobid:
        return HttpResponse('no jobid')

    tmpdir = tempdir_for_jobid(jobid)
    cancelfile = os.path.join(tmpdir, 'cancel')
    log('shard.cancel jobid', jobid)
    # FIXME - security hole.
    write_file(' ', cancelfile)
    return HttpResponse('ok')

def index(request):
    res = HttpResponse()
    backendcfg = settings.BACKEND_CONFIG % socket.gethostname()
    res['Content-type'] = 'text/plain'
    res.write('backend config file: %s\n\n' % backendcfg)
    for line in open(backendcfg):
        res.write('    %s' % line)
    return res
