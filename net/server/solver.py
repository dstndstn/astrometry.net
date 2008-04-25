#! /usr/bin/env python

import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import sys
import socket
import time
import tempfile
import thread
from datetime import datetime
from optparse import OptionParser

from urllib import urlencode
from urllib2 import urlopen

import pyfits

import astrometry.net.settings as settings

#from astrometry.net.server.log import log, reconfig_logging

hostname = socket.gethostname()
hostname = hostname.split('.')[0]
pid = os.getpid()
settings.SERVER_LOGFILE = settings.LOG_DIR + 'worker-%s-%i.log' % (hostname, pid)
print 'Logging to', settings.SERVER_LOGFILE

from astrometry.net.server.log import log


from astrometry.net.server.models import *
from astrometry.util.run_command import run_command
from astrometry.net.portal.wcs import TanWCS
from astrometry.net.portal.job import Calibration, Job

def get_header(header, key, default):
    try:
        return header[key]
    except KeyError:
        return default

#def log(*args):
#    pass

class Solver(object):
    def __init__(self, q, indexdirs):
        self.worker = Worker(queue=q)
        self.q = q
        self.worker.save()
        self.worker.start_keepalive_thread()
        self.aborting_job = False

        indexpaths = []

        for d in indexdirs:
            if not os.path.exists(d):
                print 'No such directory:', d
                continue
            files = os.listdir(d)
            for f in files:
                csuff = '.ckdt.fits'
                qsuff = '.quad.fits'
                ssuff = '.skdt.fits'
                if not f.endswith(qsuff):
                    continue
                base = os.path.join(d, f[:-len(qsuff)])
                if not (os.path.exists(base + csuff) and
                        os.path.exists(base + ssuff)):
                    continue
                print 'Found index', base
                qfile = base + qsuff
                hdus = pyfits.open(qfile)
                hdr = hdus[0].header
                indexid = get_header(hdr, 'INDEXID', None)
                hp = get_header(hdr, 'HEALPIX', -1)
                hpnside = get_header(hdr, 'HPNSIDE', 1)
                scalelo = float(get_header(hdr, 'SCALE_L', 0))
                scalehi = float(get_header(hdr, 'SCALE_U', 0))
                print 'id', indexid, 'hp', hp, 'hp nside', hpnside, 'scale [%g, %g]' % (scalelo, scalehi)
                if indexid is None:
                    continue

                (ind, nil) = Index.objects.get_or_create(indexid=indexid,
                                                         healpix=hp,
                                                         healpix_nside=hpnside,
                                                         defaults={'scalelo': scalelo,
                                                                   'scalehi': scalehi,})
                self.worker.indexes.add(ind)
                self.worker.save()
                indexpaths.append(base)

        print 'My indexes:', self.worker.pretty_index_list()

        # Write my backend.cfg file.
        (f, self.backendcfg) = tempfile.mkstemp('', 'backend.cfg-')
        os.close(f)
        f = open(self.backendcfg, 'wb')
        f.write('\n'.join(['inparallel'] +
                          ['index %s' % path for path in indexpaths]
                          ))
        f.close()

    def run_one(self):
        nextwork = self.worker.get_next_work(self.q)
        if nextwork is None:
            return False
        (qjob, work) = nextwork

        print 'Got my next job:', qjob, 'work', work

        for w in work:
            print '  index', w.index
            w.worker = self.worker
            w.inprogress = True
            w.save()

        self.worker.job = qjob
        self.worker.save()

        # check that i'm the first of my peer workers to work on this job.
        myinds = self.worker.pretty_index_list()
        for w in qjob.workers.all().exclude(id=self.worker.id):
            inds = w.pretty_index_list()
            #print 'Worker', w, 'has inds', inds
            if myinds == inds:
                if w.id < self.worker.id:
                    print 'Aborting because another Worker with the same index set is already working on this job: ', w
                    self.worker.job = None
                    self.worker.save()
                    return False

        qjob.inprogress = True
        qjob.save()

        log('Working on job', qjob)
        log('Doing work:', work)

        # retrieve the input files.
        url = qjob.get_url()
        log('Retrieving URL %s...' % url)

        job = qjob.job

        (f, axy) = tempfile.mkstemp('', 'axy-%s-' % job.jobid)
        os.close(f)
        
        fn = qjob.retrieve_to_file(axy)
        log('Saved as', fn)

        tmpdir = tempfile.mkdtemp('', 'backend-results-')
        tarfile = os.path.join(tmpdir, 'results.tar')
        backend = 'backend'
        cancelfile = os.path.join(tmpdir, 'cancel')
        # HACK - pipes?
        cmd = ('cd %(tempdir)s; %(backend)s -c %(backendconf)s -C %(cancel)s -v %(axy)s > %(logfile)s 2>&1' %
               dict(tempdir=tmpdir, backend=backend,
                    backendconf=self.backendcfg,
                    cancel=cancelfile,
                    axy=axy,
                    logfile='blind.log'))
        log('Running command', cmd)
    
        (rtn, out, err) = run_command(cmd, timeout=1,
                                      callback=lambda: self.callback(qjob, self.worker, cancelfile))

        if rtn:
            log('backend failed: rtn val %i' % rtn, ', out', out, ', err', err)

        if not self.aborting_job:
            # Send results -- only if solved??
            solvedfile = os.path.join(tmpdir, 'solved')
            if os.path.exists(solvedfile):
                log('Solved!')
                cmd = 'cd %s; tar cf %s *' % (tmpdir, tarfile)
                (rtn, out, err) = run_command(cmd)
                if rtn:
                    log('tar failed: rtn val %i' % rtn, ', out', out, ', err', err)
                url = qjob.get_put_results_url()
                f = open(tarfile, 'rb')
                tardata = f.read()
                f.close()
                log('Tardata string is %i bytes long.' % len(tardata))
                tardata = tardata.encode('base64_codec')
                log('Encoded string is %i bytes long.' % len(tardata))
                data = urlencode({ 'tar': tardata })
                log('Sending response to', url)
                log('url-encoded string is %i bytes long.' % len(data))
                f = urlopen(url, data)
                response = f.read()
                f.close()
                log('Got response:', response)

                # Remove all queued work for this job.
                qjob.work.all().delete()
                qjob.inprogress = False
                qjob.done = True
                qjob.save()

            else:
                log('Did not solve.')

                # Mark this Work as done.
                for w in work:
                    w.inprogress = False
                    w.done = True
                    w.save()

                # Check whether this is the last Work to be done.
                todo = qjob.work.all().filter(done=False)
                if todo.count() == 0:
                    qjob.inprogress = False
                    qjob.save()

                    job.set_status('Failed', 'Did not solve')
                    job.save()

        else:
            log('Job aborted.')
            self.aborting_job = False

        self.worker.job = None
        self.worker.save()

        # HACK - delete tempfiles.
        return True

    def abort_job(self, cancelfn):
        self.aborting_job = True
        log('Touching file', cancelfn)
        f = open(cancelfn, 'wb')
        f.write('x')
        f.close()

    def callback(self, qjob, worker, fn):
        js=QueuedJob.objects.all().filter(id=qjob.id)
        if js.count() == 0:
            return
        j=js[0]
        if j.done:
            self.abort_job(fn)
        # check that i'm the first of my peer workers to work on this job.
        myinds = ', '.join([str(i) for i in worker.indexes.all().order_by('indexid', 'healpix')])
        for w in j.workers.all():
            inds = ', '.join([str(i) for i in w.indexes.all().order_by('indexid', 'healpix')])
            if myinds == inds:
                if w.id < worker.id:
                    log('Aborting because another Worker with the same index set is already working on this job: ', w)
                    self.abort_job(fn)


    def run(self):
        while True:
            # abort if my Worker isn't found in the database.
            me = Worker.objects.get(id=self.worker.id)
            if not self.run_one():
                time.sleep(5)


if __name__ == '__main__':

    parser = OptionParser()

    # queue name
    # queue type
    #parser.add_option('-t', '--threads', dest='threads', type='int', default=1)

    #parser.add_option('-D', '--daemon', dest='daemon',
    #                  action='store_true', default=False)
    parser.add_option('-d', '--dir', dest='dir', default=None)

    (options, args) = parser.parse_args()

    if len(args):
        indexdirs = args
    else:
        indexdirs = [
            '/home/gmaps/INDEXES/500',
            ]

    if options.dir:
        print 'Changing to', options.dir
        os.chdir(options.dir)


    log('Test')

    (q,nil) = JobQueue.objects.get_or_create(name=settings.SITE_NAME, queuetype='solve')
    s = Solver(q, indexdirs)

    #if options.daemon:
    #    print 'Becoming daemon...'
    #    from astrometry.util.daemon import createDaemon
    #    createDaemon()

    s.run()

