import Ice
import IceGrid

#Ice.loadSlice(settings.BASEDIR + 'astrometry/net/ice/Solver.ice')

from astrometry.net.ice import SolverIce

class LoggerI(SolverIce.Logger):
    def __init__(self, logfunc):
        self.logfunc = logfunc
    def logmessage(self, msg, current=None):
        self.logfunc(msg)

class SolverClient(Ice.Application):
    def solve(self, jobid, axy, logfunc):
        comm = self.communicator()
        try:
            server = SolverIce.SolverPrx.checkedCast(comm.stringToProxy("Solver"))
        except Ice.NotRegisteredException, e:
            logfunc('Failed to find solver server:, e)
            return -1

        properties = comm.getProperties()
        adapter = comm.createObjectAdapter('Callback.Client')
        myid = comm.stringToIdentity('callbackReceiver')
        adapter.add(LoggerI(logfunc), myid)
        adapter.activate()
        myproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))
        tardata = server.solve(jobid, axy, myproxy)
        return tardata
