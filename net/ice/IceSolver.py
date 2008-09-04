#! /usr/bin/env python

import sys
import time

import Ice
import IceGrid

import SolverIce

from astrometry.util.file import *

theice = None
configfile = 'config.client'

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


class LoggerI(SolverIce.Logger):
    def __init__(self, logfunc):
        self.logfunc = logfunc
    def logmessage(self, msg, current=None):
        self.logfunc(msg)

class SolverResult(object):
    def __init__(self):
        self.tardata = None
        self.failed = False
    def ice_response(self, tardata):
        print 'async response'
        self.tardata = tardata
    def ice_exception(self, ex):
        print 'async exception:', ex
        self.failed = True
    def isdone(self):
        return self.failed or self.tardata is not None

def solve(jobid, axy, logfunc):
    ice = get_ice()

    # IceGrid::Query findAllObjectsByType
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
        #server = ice.stringToProxy(s)
        #server = SolverIce.SolverPrx.checkedCast(server)
        server = SolverIce.SolverPrx.checkedCast(s)
        servers.append(server)
        
    # FIXME -- may need to create one callback proxy per server.
    props = ice.getProperties()
    adapter = ice.createObjectAdapter('Callback.Client')
    myid = ice.stringToIdentity('callbackReceiver')
    adapter.add(LoggerI(logfunc), myid)
    adapter.activate()
    logproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))

    results = [SolverResult() for s in solvers]

    for (s,r) in zip(solvers,results):
        server.solve_async(r, jobid, axy, logproxy)

    waiting = [r for r in results]
    tardata = None
    while len(waiting):
        time.sleep(1)
        for r in waiting:
            if r.isdone():
                if r.tardata is not None:
                    tardata = r.tardata
                    break
        waiting = [r for r in waiting if not r.isdone()]

    if tardata is None:
        print 'all servers failed.'

    #tardata = server.solve(jobid, axy, logproxy)
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
        print msg

    tardata = solve(jobid, axydata, logfunc)

    print 'got %i bytes of tardata.' % len(tardata)

