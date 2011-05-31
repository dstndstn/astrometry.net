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
import subprocess
import signal

from StringIO import StringIO

import Ice

import SolverIce
from astrometry.util.file import *

import logging
import socket
logfile = 'logs/solver-%s.log' % (socket.gethostname().split('.')[0])
logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=logfile)
def logmsg(*msg):
    logging.debug(' '.join([str(m).decode('latin_1', 'backslashreplace') for m in msg]))

_backend = None
def get_backend_lib():
    global _backend
    if _backend:
        return _backend
    _libname = ctypes.util.find_library('libbackend.so')
    if _libname:
        _backend = ctypes.CDLL(_libname)
    else:
        # p = os.path.join(os.path.dirname(__file__), 'libbackend.so')
        # FIXME
        p = '/data1/dstn/dsolver/astrometry/blind/libbackend.so'
        _backend = ctypes.CDLL(p)

    return _backend

class SolverI(SolverIce.Solver):
    def __init__(self, name, scale):
        self.name = name
        self.scale = scale

        # The libbackend.so library
        self.Backend = None
        # This object's backend object
        self.backend = None

        print 'SolverServer running: pid', os.getpid()
        # HACK
        self.dirs = {}

        ### If running ctypes version:
        print 'SolverServer initializing backend...'
        self.init_backend()
        print 'SolverServer initialized backend.'

    def init_backend(self):
        Backend = get_backend_lib()
		log_level ll = 3;
        Backend.log_init(ll);
        Backend.log_set_thread_specific();

        configfn = self.get_config_file();

        backend = Backend.backend_new()
        logmsg('Reading config file ', configfn)
        if Backend.backend_parse_config_file(backend, configfn):
            print 'Failed to initialize backend.'
            sys.exit(-1)
        self.Backend = Backend
        self.backend = backend

    def solve_ctypes(self, jobid, axy, logger, axyfn, cancelfn, solvedfn, mydir, current=None):
        if self.backend is None:
            self.init_backend()
        backend = self.backend
        Backend = self.Backend

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

        job = Backend.backend_read_job_file(backend, axyfn)
        #print 'job is 0x%x' % job
        if not job:
            print 'Failed to read job.'
            return
        Backend.job_set_base_dir(job, mydir)
        Backend.job_set_cancel_file(job, cancelfn)
        Backend.job_set_solved_file(job, solvedfn)
        Backend.backend_run_job(backend, job)
        Backend.job_free(job)

    def solve_subprocess(self, jobid, axy, logger, axyfn, cancelfn, solvedfn, mydir, current=None):
        configfn = self.get_config_file()
        logmsg('Solving jobid ', jobid)
        t0 = time.time()
        myname = socket.gethostname().split('.')[0]

        def pipe_log_messages(f, logger):
            import select
            fno = f.fileno()
            while True:
                time.sleep(1.)
                try:
                    (ready, nil1, nil2) = select.select([fno], [], [], 0.)
                    if not len(ready):
                        continue
                    s = os.read(fno, 1000000)
                    if len(s) == 0:
                        # eof
                        break
                    logmsg('t', (time.time()-t0), 'piping log messages (len %i)' % len(s))
                    logger.logmessage(myname + ':\n' + s)
                except Exception, e:
                    logmsg('t', (time.time()-t0), 'error:', e)
                    break

        command = ('/data1/dstn/dsolver/astrometry/blind/backend ' +
                   '-v -c %s -d %s -C %s -s %s %s') % (configfn, mydir, cancelfn, solvedfn, axyfn)
        logmsg('Running command:', command)
        sub = subprocess.Popen(command, bufsize=1, shell=True, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, cwd=mydir)
        (childin, childouterr) = (sub.stdin, sub.stdout)
        childin.close()
        pipe_log_messages(childouterr, logger)
        logmsg('Solving command returned.')
        if sub.poll() is None:
            # pipe_log_messages bailed: probably client hung up on us.  kill sub.
            os.kill(sub.pid, signal.SIGTERM)
            time.sleep(1.)
        logmsg('Solving command return value: ', sub.poll())

    def get_config_file(self):
        configfn = '/data1/dstn/dsolver/backend-config/backend-scale%i.cfg' % self.scale
        return configfn

    def solve(self, jobid, axy, logger, current=None):
        logmsg(self.name + ' got a solve request.')
        logmsg('jobid', jobid, 'axy has length', len(axy))

        logger.logmessage('Hello from %s' % self.name)
        hostname = socket.gethostname().split('.')[0]
        #print 'I am host', hostname
        mydir = tempfile.mkdtemp('', 'backend-'+jobid+'-')

        self.dirs[jobid] = mydir

        logmsg('Working in temp directory', mydir)
        axyfn = os.path.join(mydir, 'job.axy')
        write_file(axy, axyfn)

        #jid = jobid.replace('/', '-')
        #cancelfn = '/tmp/%s.cancel' % (jid)
        #solvedfn = '/tmp/%s.solved' % (jid)

        cancelfn = mydir + '/cancel'
        solvedfn = mydir + '/solved'

        self.solve_ctypes(jobid, axy, logger, axyfn, cancelfn, solvedfn, mydir, current)
        #self.solve_subprocess(jobid, axy, logger, axyfn, cancelfn, solvedfn, mydir, current)

        solved = os.path.exists(mydir + '/wcs.fits')
        if solved:
            # tell the other servers to stop...
            #write_file('', cancelfn)
            #write_file('', solvedfn)
            logmsg('Solved.')
        else:
            logmsg('Did not solve.')

        (sin,sout) = os.popen2('cd %s; tar c .' % (mydir))
        sin.close()
        tardata = sout.read()
        sout.close()
        return (tardata, solved)

    def cancel(self, jobid, current=None):
        if not jobid in self.dirs:
            logmsg("Request to cancel a job I'm not working on: " + jobid)
            return
        mydir = self.dirs[jobid]
        cancelfn = mydir + '/cancel'
        write_file('', cancelfn)
        logmsg('Cancelled job ' + jobid)

    def status(self, current=None):
        configfn = self.get_config_file()
        return 'config file: %s' % configfn

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
