#! /usr/bin/env python

import sys
import time

import Glacier2
import Ice
import IceGrid

import SolverIce

from astrometry.util.file import *
import astrometry.net.settings as settings

theice = None
# FIXME
configfile = settings.BASEDIR + 'astrometry/net/ice/config.client'

def initIce():
    global theice
    settings = Ice.InitializationData()
    settings.properties = Ice.createProperties(None, settings.properties)
    settings.properties.load(configfile)
    theice = Ice.initialize(settings)

def get_ice():
    if not theice:
        initIce()
    return theice

def get_router_session(ice):
    router = theice.getDefaultRouter()
    router = Glacier2.RouterPrx.checkedCast(router)
    if not router:
        print 'not a glacier2 router'
        return -1
    session = None
    try:
        session = router.createSession('test', 'test')
    except Glacier2.PermissionDeniedException,ex:
        print 'router session permission denied:', ex
    return (router, session)


class LoggerI(SolverIce.Logger):
    def __init__(self, logfunc):
        self.logfunc = logfunc
    def logmessage(self, msg, current=None):
        self.logfunc(msg)

class SolverResult(object):
    def __init__(self):
        self.tardata = None
        self.failed = False
        self.solved = False
    def ice_response(self, tardata, solved):
        print 'async response: ', (solved and 'solved' or 'did not solve')
        self.tardata = tardata
        self.solved = solved
    def ice_exception(self, ex):
        print 'async exception:', ex
        self.failed = True
    def isdone(self):
        return self.failed or self.tardata is not None

def solve(jobid, axy, logfunc):
    ice = get_ice()

    (router, session) = get_router_session(ice)
    category = router.getCategoryForClient()

    q = ice.stringToProxy('SolverIceGrid/Query')
    q = IceGrid.QueryPrx.checkedCast(q)
    print 'q is', q
    solvers = q.findAllObjectsByType('::SolverIce::Solver')
    print 'Found %i solvers' % len(solvers)
    for s in solvers:
        print '  ', s

    servers = []
    for s in solvers:
        print 'Resolving ', s
        server = SolverIce.SolverPrx.checkedCast(s)
        servers.append(server)

    props = ice.getProperties()
    adapter = ice.createObjectAdapter('Callback.Client')
    myid = ice.stringToIdentity('callbackReceiver')
    myid.category = category
    adapter.add(LoggerI(logfunc), myid)
    adapter.activate()
    logproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))

    results = [SolverResult() for s in solvers]

    for (s,r) in zip(servers,results):
        s.solve_async(r, jobid, axy, logproxy)

    waiting = [r for r in results]
    tardata = None
    while len(waiting):
        time.sleep(1)
        for r in waiting:
            if r.isdone():
                if r.tardata is not None and r.solved:
                    tardata = r.tardata
                    break
        if tardata:
            break
        waiting = [r for r in waiting if not r.isdone()]
    return tardata



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: %s <jobid> <job.axy>' % sys.argv[0]
        sys.exit(-1)

    jobid = sys.argv[1]
    axyfn = sys.argv[2]
    axydata = read_file(axyfn)
    print 'jobid is ', jobid
    print 'axyfile is %i bytes' % len(axydata)

    def logfunc(msg):
        print msg,

    tardata = solve(jobid, axydata, logfunc)

    if tardata is None:
        print 'got no tardata'
    else:
        print 'got %i bytes of tardata.' % len(tardata)
        # extract tardata into this directory.
        p = os.popen('tar x', 'w')
        p.write(tardata)
        p.close()

