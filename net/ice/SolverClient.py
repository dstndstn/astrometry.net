import sys

import Ice

import SolverIce

theice = None
configfile = 'config.client'

def initIce():
    global theice
    settings = Ice.InitializationData()
    settings.properties = createProperties(None, settings.properties)
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
    #axydata = read_file(job.get_axy_filename())
    
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

