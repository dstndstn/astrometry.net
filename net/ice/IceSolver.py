#! /usr/bin/env python

import sys

import Ice

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


def solve(jobid, axy, logfunc):
    ice = get_ice()
    server = ice.stringToProxy('Solver')
    server = SolverIce.SolverPrx.checkedCast(server)
    props = ice.getProperties()
    adapter = ice.createObjectAdapter('Callback.Client')
    myid = ice.stringToIdentity('callbackReceiver')
    adapter.add(LoggerI(logfunc), myid)
    adapter.activate()
    logproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))
    tardata = server.solve(jobid, axy, logproxy)
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

