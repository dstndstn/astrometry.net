import sys
import Ice
import Glacier2
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
    def __init__(self, jobid, axy, logfunc):
        self.jobid = jobid
        self.axy = axy
        self.logfunc = logfunc
        
    #def solve(self, jobid, axy, logfunc):
    def run(self, args):
        jobid = self.jobid
        axy = self.axy
        logfunc = self.logfunc

        comm = self.communicator()
        print 'comm is', comm

        print 'creating session with Glacier2...'
        router = comm.getDefaultRouter()
        if not router:
            print 'no router.'
            return -1
        router = Glacier2.RouterPrx.checkedCast(router)
        if not router:
            print 'not a glacier2 router'
            return -1
        try:
            router.createSession('test', 'test')
        except Glacier2.PermissionDeniedException,ex:
            print 'permission denied:', ex

        try:
            p = comm.stringToProxy('Solver')
            print 'proxy is', p
            server = SolverIce.SolverPrx.checkedCast(p)
            print 'server is', server
        except Ice.NotRegisteredException, e:
            logfunc('Failed to find solver server:', e)
            return -1

        category = router.getCategoryForClient()

        properties = comm.getProperties()
        adapter = comm.createObjectAdapter('Callback.Client')
        myid = comm.stringToIdentity('callbackReceiver')
        myid.category = category
        adapter.add(LoggerI(logfunc), myid)
        adapter.activate()
        myproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))
        tardata = server.solve(jobid, axy, myproxy)
        return tardata


if __name__ == '__main__':
    def testlogfunc(msg):
        print msg
    c = SolverClient('testjobid', 'XXXXXXXXXX', testlogfunc)
    #c.solve()
    sys.exit(c.main(sys.argv, 'config.client'))
