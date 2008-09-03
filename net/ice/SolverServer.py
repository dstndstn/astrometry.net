#!/usr/bin/env python

import sys
import traceback
import time
import socket
import tempfile
import ctypes
import ctypes.util
import os.path

import Ice

import SolverIce
from astrometry.util.file import *

_backend = None
_libname = ctypes.util.find_library('libbackend.so')
if _libname:
    _backend = ctypes.CDLL(_libname)
else:
    #p = os.path.join(os.path.dirname(__file__), 'libbackend.so')
    # FIXME
    p = '/data1/dstn/dsolver/astrometry/blind/libbackend.so'
    _backend = ctypes.CDLL(p)


class SolverI(SolverIce.Solver):
    def __init__(self, name):
        self.name = name

    def solve(self, jobid, axy, logger, current=None):
        print self.name + ' got a solve request.'
        print 'jobid', jobid, 'axy has length', len(axy)
        logger.logmessage('Hello logger.')

        #time.sleep(1)
        #logger.logmessage('Hello again.')
        #time.sleep(1)

        hostname = socket.gethostname().split('.')[0]
        print 'I am host', hostname
        configfn = '/data1/dstn/dsolver/backend-config/backend-%s.cfg' % hostname
        print 'Reading config file', configfn
        mydir = tempfile.mkdtemp()
        print 'Working in temp directory', mydir
        axyfn = os.path.join(mydir, 'job.axy')
        write_file(axy, axyfn)

        # BUG - should do this once, outside this func!
        Backend = _backend
        Backend.log_init(3)
        Backend.log_set_thread_specific()
        backend = Backend.backend_new()
        if Backend.backend_parse_config_file(backend, configfn):
            print 'Failed to initialize backend.'
            sys.exit(-1)
        #backend.log_to_fd(f.fileno())
        job = Backend.backend_read_job_file(backend, axyfn)
        if not job:
            print 'Failed to read job.'
            return
        Backend.job_set_base_dir(job, mydir)
        #backend.job_set_cancel_file(job, cancelfile)
        Backend.backend_run_job(backend, job)
        Backend.job_free(job)

        tardata = 'Goodbye, all done here.'
        return tardata
    
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
