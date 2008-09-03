#!/usr/bin/env python

import sys
import traceback
import time
import Ice

#Ice.loadSlice(settings.BASEDIR + 'astrometry/net/ice/Solver.ice')
#Ice.loadSlice('Hello.ice')
#from astrometry.net.ice import SolverIce
import SolverIce

class SolverI(SolverIce.Solver):
    def __init__(self, name):
        self.name = name

    def solve(self, jobid, axy logger, current=None):
        print self.name + ' got a solve request.'
        print 'jobid', jobid, 'axy has length', len(axy)
        logger.logmessage('Hello logger.')
        time.sleep(1)
        logger.logmessage('Hello again.')
        time.sleep(1)
        return 'Goodbye, all done here.'
    
    def shutdown(self, current=None):
        print self.name + " shutting down..."
        current.adapter.getCommunicator().shutdown()

class Server(Ice.Application):
    def run(self, args):
        properties = self.communicator().getProperties()
        adapter = self.communicator().createObjectAdapter("Solver")
        myid = self.communicator().stringToIdentity(properties.getProperty("Identity"))
        print 'myid is', myid
        print 'programname is', properties.getProperty("Ice.ProgramName")
        adapter.add(SolverI(properties.getProperty("Ice.ProgramName")), myid)
        adapter.activate()
        self.communicator().waitForShutdown()
        return 0

if __name__ == '__main__':
    app = Server()
    sys.exit(app.main(sys.argv, 'config.grid'))
