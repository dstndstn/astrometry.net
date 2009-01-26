#! /usr/bin/env python

import sys
import time
import threading

import Glacier2
import Ice
import IceGrid

import SolverIce

from astrometry.util.file import *
import astrometry.net.settings as settings
from astrometry.net.portal.log import log

# FIXME
configfile = settings.BASEDIR + 'astrometry/net/ice/config.client'

username = 'foo'
password = 'bar'

printlog = False
def logmsg(*msg):
    if printlog:
        print ' '.join([str(m) for m in msg])
    else:
        log('IceSolver:', *msg)

theclient = None
theclientlock = threading.Lock()

class SolverClient(object):
    def __init__(self):
        pass

    def init(self):
        settings = Ice.InitializationData()
        settings.properties = Ice.createProperties(None, settings.properties)
        settings.properties.load(configfile)
        ice = Ice.initialize(settings)
        self.ice = ice

        logmsg('get session')
        router = ice.getDefaultRouter()
        router = Glacier2.RouterPrx.checkedCast(router)
        if not router:
            logmsg('not a glacier2 router')
            raise 'not a glacier2 router'
        session = None
        try:
            session = router.createSession(username, password) #'test-%i' % int(time.time()), 'test')
        except Glacier2.PermissionDeniedException,ex:
            logmsg('router session permission denied:', ex)
            raise ex
        except Glacier2.CannotCreateSessionException,ex:
            logmsg('router session: cannot create:', ex)
            raise ex
        logmsg('router is', router)
        logmsg('session is', session)
        self.router = router
        self.session = session

        logmsg('creating adapter...')
        self.adapter = ice.createObjectAdapter('Callback.Client')
        logmsg('created adapter', self.adapter)
        self.adapter.activate()


    # this will be called from multiple threads.
    def solve(self, jobid, axy, logfunc):
        category = self.router.getCategoryForClient()
        myid = Ice.Identity()
        myid.category = category
        myid.name = Ice.generateUUID()
        logmsg('my id:', myid)

        self.adapter.add(LoggerI(logfunc), myid)
        logproxy = SolverIce.LoggerPrx.uncheckedCast(self.adapter.createProxy(myid))
        logmsg('my logger:', logproxy)

        logmsg('get servers')
        servers = self.find_all_solvers()
        logmsg('servers:', servers)

        results = [SolverResult(s) for s in servers]

        logmsg('making ICE calls...')
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

        logmsg('all ICE calls have returned')

        # grace period to let servers send their last log messages.
        time.sleep(3)
        return tardata

    def find_all_solvers(self):
        q = self.ice.stringToProxy('SolverIceGrid/Query')
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
        #logmsg('logger callback')
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
    global theclient
    
    theclientlock.acquire()
    if not theclient:
        theclient = SolverClient()
        theclient.init()
    theclientlock.release()

    tardata = theclient.solve(jobid, axy, logfunc)

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

