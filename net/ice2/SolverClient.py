import sys
import Ice
#import IceGrid
#Ice.loadSlice(settings.BASEDIR + 'astrometry/net/ice/Solver.ice')
#Ice.loadSlice('Solver.ice')

import SolverIce

#from astrometry.net.ice import SolverIce


class LoggerI(SolverIce.Logger):
    def __init__(self, logfunc):
        self.logfunc = logfunc
    def logmessage(self, msg, current=None):
        self.logfunc(msg)

class SolverClient(Ice.Application):
    def __init__(self, axy, logfunc):
        self.axy = axy
        self.logfunc = logfunc
        
    #def solve(self, jobid, axy, logfunc):
    def run(self, args):
        axy = self.axy
        logfunc = self.logfunc

        comm = self.communicator()
        print 'comm is', comm

        try:
            p = comm.stringToProxy('Solver')
            print 'proxy is', p
            server = SolverIce.SolverPrx.checkedCast(p)
            print 'server is', server
        except Ice.NotRegisteredException, e:
            logfunc('Failed to find solver server:', e)
            return -1

        properties = comm.getProperties()
        adapter = comm.createObjectAdapter('Callback.Client')
        myid = comm.stringToIdentity('callbackReceiver')
        adapter.add(LoggerI(logfunc), myid)
        adapter.activate()
        myproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))
        tardata = server.solve(axy, myproxy)
        return tardata


if __name__ == '__main__':
    def testlogfunc(msg):
        print msg
    c = SolverClient('/path/to/axy', testlogfunc)
    #c.solve()
    sys.exit(c.main(sys.argv, 'config.client'))
