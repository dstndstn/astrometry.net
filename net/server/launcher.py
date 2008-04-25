import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import time
import datetime

import astrometry.net.settings as settings

from astrometry.net.server.models import *
from astrometry.net.server.indexes import load_indexes

from astrometry.util.run_command import run_command

load_indexes()

# 41 MB.
i510up = list(Index.objects.all().filter(indexid__gte=510))

# 1.2 GB.
i505to9 = list(Index.objects.all().filter(indexid__range=(505, 509)))

# 1.3 GB
i504 = list(Index.objects.all().filter(indexid=504))

# 2.6 GB
i503 = list(Index.objects.all().filter(indexid=503))

# 5.2 GB
i502 = list(Index.objects.all().filter(indexid=502))

def launch_cslab(args, inds):
    print 'launching cslab:', args
    print 'indexes:', inds
    (hname,) = args
    cmd = 'echo "%s" | ssh -x -T solver-cslab' % hname
    print 'Running:', cmd
    os.system(cmd)
    print 'launch command returned.'

    #(rtn, out, err) = run_command(cmd)
    #if rtn:
    #    print 'Command failed: rtn %i' % rtn
    #print 'Out:', out
    #print 'Err:', err

def launch_oven(args, inds):
    print 'launching oven:', args
    print 'indexes:', inds
    (hname,) = args
    cmd = '%ssimple-daemon %s' % (settings.WEB_DIR + 'execs/', settings.WEB_DIR + 'server/run-solver-oven.sh')
    print 'Running:', cmd
    os.system(cmd)
    print 'launch command returned.'

hosts = [
    (launch_oven,  ('oven',     ), i510up,  4),
    #(launch_cslab, ('cluster60',), i505to9, 2),
    (launch_cslab, ('cluster59',), i505to9, 2),
    (launch_cslab, ('cluster58',), i504,    2),
    (launch_cslab, ('cluster57',), i504,    2),
    (launch_cslab, ('cluster56',), i503,    2),
    (launch_cslab, ('cluster55',), i503,    2),
    (launch_cslab, ('cluster54',), i502,    2),
    (launch_cslab, ('cluster53',), i502,    2),
    ]

Ntarget = 4

# FIXME - this should be per-queue...

# FIXME - this doesn't work quite right if a single machine is listed
# with multiple index sets.

def launch_solvers():
    # check that each Index is loaded by enough Workers.
    tolaunch = {}
    for ind in Index.objects.all().order_by('indexid', 'healpix'):
        N = ind.workers.all().count()
        if N >= Ntarget:
            continue
        #print 'Index %s has only %i workers.' % (ind, N)
        for i, (cmd, args, hostinds, Nmax) in enumerate(hosts):
            # does this host have this index?
            if not ind in hostinds:
                continue
            # is this host already running the maximum number of
            # instances?
            (hname,) = args
            NW = Worker.objects.all().filter(hostname__istartswith=hname).count()
            if NW >= Nmax:
                continue
            Nlaunch = min(Ntarget - NW, Nmax)
            #print 'Launch %i new worker(s) on %s for index %s' % (Nlaunch, hname, ind)
            if not i in tolaunch:
                tolaunch[i] = []
            tolaunch[i] += [ind]

    for i,inds in tolaunch.items():
        host = hosts[i]
        (cmd, args, hostinds, nmax) = host
        (hname,) = args
        print 'Host', hname, ': launch', inds
        cmd(args, inds)


if __name__ == '__main__':
    while True:
        print 
        print 'Checking if any solvers need to be launched...'
        print
        launch_solvers()
        print 
        print 'Sleeping...'
        print
        time.sleep(30)

