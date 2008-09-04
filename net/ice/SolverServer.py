#!/usr/bin/env python

import sys
import traceback
import time
import socket
import tempfile
import ctypes
import ctypes.util
import os
import os.path
import thread
import fcntl
import tarfile
from StringIO import StringIO

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
    def __init__(self, name, scale):
        self.name = name
        self.scale = scale
        print 'SolverServer running: pid', os.getpid()

    def solve_ctypes(self, jobid, axy, logger, configfn, axyfn, mydir, current=None):
        # BUG - should do this once, outside this func!
        Backend = _backend
        Backend.log_init(3)
        Backend.log_set_thread_specific()

        def pipe_log_messages(p, logger):
            fcntl.fcntl(p, fcntl.F_SETFL, os.O_NDELAY | os.O_NONBLOCK)
            f = os.fdopen(p)
            while not f.closed:
                try:
                    s = f.read()
                    print 'piping log messages:', s
                    logger.logmessage(s)
                except IOError, e:
                    if e.errno != 11:
                        print 'io error:', e
                time.sleep(1.)

        (rpipe,wpipe) = os.pipe()
        Backend.log_to_fd(wpipe)
        thread.start_new_thread(pipe_log_messages, (rpipe, logger))

        backend = Backend.backend_new()
        #print 'backend is 0x%x' % backend
        if Backend.backend_parse_config_file(backend, configfn):
            print 'Failed to initialize backend.'
            sys.exit(-1)
        job = Backend.backend_read_job_file(backend, axyfn)
        #print 'job is 0x%x' % job
        if not job:
            print 'Failed to read job.'
            return
        Backend.job_set_base_dir(job, mydir)
        #Backend.job_set_cancel_file(job, cancelfile)
        Backend.backend_run_job(backend, job)
        Backend.job_free(job)

    def solve_subprocess(self, jobid, axy, logger, configfn, axyfn, mydir, current=None):

        def pipe_log_messages(f, logger):
            fcntl.fcntl(f.fileno(), fcntl.F_SETFL, os.O_NDELAY | os.O_NONBLOCK)
            while not f.closed:
                try:
                    s = f.read()
                    #print 'piping log messages:', s
                    print 'piping log messages (len %i)' % len(s)
                    logger.logmessage(s)
                except IOError, e:
                    if e.errno != 11:
                        print 'io error:', e
                except Exception, e:
                    print 'other error:', e
                    break
                time.sleep(1.)

        cancelfn = '%s/%s.cancel' % (mydir, jobid)
        command = ('cd %s; /data1/dstn/dsolver/astrometry/blind/backend ' +
                   '-v -c %s -d %s -C %s %s') % (mydir, configfn, mydir, cancelfn, axyfn)
        (childin, childouterr) = os.popen4(command)
        childin.close()
        pipe_log_messages(childouterr, logger)

    def solve(self, jobid, axy, logger, current=None):
        print self.name + ' got a solve request.'
        print 'jobid', jobid, 'axy has length', len(axy)
        logger.logmessage('Hello from %s' % self.name)
        hostname = socket.gethostname().split('.')[0]
        print 'I am host', hostname
        configfn = '/data1/dstn/dsolver/backend-config/backend-scale%i.cfg' % self.scale
        print 'Reading config file', configfn
        mydir = tempfile.mkdtemp()
        print 'Working in temp directory', mydir
        axyfn = os.path.join(mydir, 'job.axy')
        write_file(axy, axyfn)

        # self.solve_ctypes(jobid, axy, logger, configfn, axyfn, mydir, current)
        self.solve_subprocess(jobid, axy, logger, configfn, axyfn, mydir, current)

        (sin,sout) = os.popen2('cd %s; tar c .' % (mydir))
        sin.close()
        tardata = sout.read()
        sout.close()
        return tardata
    
        #files = os.listdir(mydir)
        #f = StringIO()
        #tar = tarfile.open(name='', mode='w', fileobj=f)
        #tar.close()
        #tardata = f.getvalue()
        #f.close()
        #return tardata


    def shutdown(self, current=None):
        print self.name + " shutting down..."
        current.adapter.getCommunicator().shutdown()

class Server(Ice.Application):
    def __init__(self, scale):
        self.scale = scale
    def run(self, args):
        ice = self.communicator()
        properties = ice.getProperties()
        adapter = ice.createObjectAdapter("OneSolver")
        myid = ice.stringToIdentity(properties.getProperty("Identity"))
        print 'myid is', myid
        progname = properties.getProperty("Ice.ProgramName")
        print 'programname is', progname
        adapter.add(SolverI(progname, scale), myid)
        adapter.activate()
        ice.waitForShutdown()
        return 0

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        s = SolverI('tester')
        class MyLogger(SolverIce.Logger):
            def logmessage(self, msg):
                print msg,
        s.solve('fake-jobid', read_file('job.axy'), MyLogger())
        sys.exit(0)
    print 'SolverServer.py args:', sys.argv
    scale = int(sys.argv[1])
    print 'scale %i' % scale
    app = Server(scale)
    sys.exit(app.main(sys.argv, 'config.grid'))
