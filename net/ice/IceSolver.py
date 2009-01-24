#! /usr/bin/env python

import sys
import time

import Glacier2
import Ice
import IceGrid

import SolverIce

from astrometry.util.file import *
import astrometry.net.settings as settings
from astrometry.net.portal.log import log

theice = None
# FIXME
configfile = settings.BASEDIR + 'astrometry/net/ice/config.client'

printlog = False
def logmsg(*msg):
    if printlog:
        print ' '.join([str(m) for m in msg])
    else:
        log(msg)
        
    

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
        logmsg('not a glacier2 router')
        return -1
    session = None
    try:
        session = router.createSession('test-%i' % int(time.time()), 'test')
    except Glacier2.PermissionDeniedException,ex:
        logmsg('router session permission denied:', ex)
    except Glacier2.CannotCreateSessionException,ex:
        logmsg('router session: cannot create:', ex)
    return (router, session)

def find_all_solvers(ice):
    q = ice.stringToProxy('SolverIceGrid/Query')
    q = IceGrid.QueryPrx.checkedCast(q)
    solvers = q.findAllObjectsByType('::SolverIce::Solver')
    logmsg('Found %i solvers' % len(solvers))
    for s in solvers:
        logmsg('  ', s)

    servers = []
    for s in solvers:
        logmsg('Resolving ', s)
        server = SolverIce.SolverPrx.checkedCast(s)
        servers.append(server)

    return servers

class LoggerI(SolverIce.Logger):
    def __init__(self, logfunc):
        self.logfunc = logfunc
    def logmessage(self, msg, current=None):
        self.logfunc(msg)

class SolverResult(object):
    def __init__(self, server):
        self.tardata = None
        self.failed = False
        self.solved = False
        self.server = server
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

    servers = find_all_solvers(ice)

    props = ice.getProperties()
    adapter = ice.createObjectAdapter('Callback.Client')
    myid = ice.stringToIdentity('callbackReceiver')
    category = router.getCategoryForClient()
    myid.category = category
    adapter.add(LoggerI(logfunc), myid)
    adapter.activate()
    logproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))

    results = [SolverResult(s) for s in servers]

    for (s,r) in zip(servers,results):
        s.solve_async(r, jobid, axy, logproxy)

    waiting = [r for r in results]
    tardata = None

    lastping = time.time()
    pingperiod = 30 # seconds

    while len(waiting):
        time.sleep(1)
        for r in waiting:
            if r.isdone():
                if r.tardata is not None and r.solved:
                    tardata = r.tardata
                    break
        if tardata:
            for r in waiting:
                if not r.isdone():
                    logfunc('Cancelling ' + str(r.server))
                    r.server.cancel(jobid)
            break
        waiting = [r for r in waiting if not r.isdone()]

        tnow = time.time()
        if tnow - lastping > pingperiod:
            logfunc('Sending pings...')
            for r in waiting:
                r.server.ice_oneway().ice_ping()
            lastping = tnow

    # grace period to let servers send their last log messages.
    time.sleep(3)

    return tardata

def status():
    ice = get_ice()
    (router, session) = get_router_session(ice)
    servers = find_all_solvers(ice)
    print 'Found %i servers.' % len(servers)
    for s in servers:
        print 'Getting status for', s
        st = s.status()
        print 'Got status:', st
    

if __name__ == '__main__':
    printlog = True
    if len(sys.argv) == 2 and sys.argv[1] == 'status':
        status()
        sys.exit(0)

    if len(sys.argv) != 3:
        print 'Usage: %s <jobid> <job.axy>\n' % sys.argv[0]
        print '   or: %s status\n' % sys.argv[0]
        sys.exit(-1)

    jobid = sys.argv[1]
    axyfn = sys.argv[2]
    axydata = read_file(axyfn)
    print 'jobid is ', jobid
    print 'axyfile is %i bytes' % len(axydata)

    def logfunc(msg):
        logmsg(msg)

    tardata = solve(jobid, axydata, logfunc)

    if tardata is None:
        print 'got no tardata'
    else:
        print 'got %i bytes of tardata.' % len(tardata)
        # extract tardata into this directory.
        p = os.popen('tar x', 'w')
        p.write(tardata)
        p.close()

