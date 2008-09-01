#! /usr/bin/env python

from watcher_common import *
from astrometry.net import settings

import Ice, IceGrid

Ice.loadSlice(settings.BASEDIR + 'astrometry/net/ice/Solver.ice')
from astrometry.net.ice import SolverIce

blindlog=None

def userlog(msg):
    f = open(blindlog, 'a')
    f.write(' '.join(map(str, msg)) + '\n')
    f.close()

class LoggerI(SolverIce.Logger):
    def logmessage(self, msg, current=None):
        userlog(msg)

class SolverClient(Ice.Application):
    def solve(self, jobid, axy):
        comm = self.communicator()
        try:
            server = SolverIce.SolverPrx.checkedCast(comm.stringToProxy("Solver"))
        except Ice.NotRegisteredException, e:
            logmsg('Failed to find solver server:, e)
            log('Failed to find solver server:, e)
            return -1

        properties = comm.getProperties()
        adapter = comm.createObjectAdapter("Callback.Client")
        myid = comm.stringToIdentity('callbackReceiver')
        adapter.add(LoggerI(), myid)
        adapter.activate()
        myproxy = SolverIce.LoggerPrx.uncheckedCast(adapter.createProxy(myid))

        tardata = server.solve(jobid, axy, myproxy)
        return tardata

class WatcherIce(Watcher):
    def solve_job(self, job):
        global blindlog
        blindlog = job.get_filename('blind.log')
        axy = read_file(job.get_axy_filename())

        client = SolverClient()
        tardata = client.solve(job.jobid, axy)
        return tardata

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s <input-file>' % sys.argv[0]
        sys.exit(-1)
    joblink = sys.argv[1]
    w = WatcherIce()
    sys.exit(w.main(joblink))

