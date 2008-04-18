import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import sys
import socket
import time
import tempfile
import thread
from datetime import datetime

from urllib import urlencode
from urllib2 import urlopen

import pyfits

import astrometry.net.settings as settings

settings.SERVER_LOGFILE = 'worker.log'

from astrometry.net.server.log import log
from astrometry.net.server.models import *
from astrometry.net.util.run_command import run_command

def get_header(header, key, default):
    try:
        return header[key]
    except KeyError:
        return default

class Solver(object):
    def __init__(self, q, indexdirs):
        self.worker = Worker(queue=q)
        self.q = q
        #self.indexdirs = indexdirs
        self.worker.save()
        self.worker.start_keepalive_thread()

        self.indexes = []
        for d in indexdirs:
            if not os.path.exists(d):
                print 'No such directory:', d
                continue
            files = os.listdir(d)
            for f in files:
                csuff = '.ckdt.fits'
                qsuff = '.quad.fits'
                ssuff = '.skdt.fits'
                if f.endswith(csuff):
                    base = os.path.join(d, f[:-len(csuff)])
                    if (os.path.exists(base + qsuff) and
                        os.path.exists(base + ssuff)):
                        print 'Found index', base
                        qfile = base + qsuff
                        hdus = pyfits.open(qfile)
                        hdr = hdus[0].header
                        indexid = get_header(hdr, 'INDEXID', None)
                        hp = get_header(hdr, 'HEALPIX', -1)
                        hpnside = get_header(hdr, 'HPNSIDE', 1)
                        print 'id', indexid, 'hp', hp, 'hp nside', hpnside
                        if indexid is not None:
                            self.indexes.append((base, indexid, hp, hpnside))

        for (fn, indexid, hp, hpnside) in self.indexes:
            (ind,nil) = Index.objects.get_or_create(indexid=indexid,
                                                    healpix=hp,
                                                    healpix_nside=hpnside)
            self.worker.indexes.add(ind)
        self.worker.save()

        print 'My indexes:', self.worker.pretty_index_list()

        # Write my backend.cfg file.
        (f, self.backendcfg) = tempfile.mkstemp('', 'backend.cfg-')
        os.close(f)
        f = open(self.backendcfg, 'wb')
        f.write('\n'.join(['inparallel'] +
                          ['index %s' % path for (path, indid, hp, hpnside) in self.indexes]
                          ))
        f.close()

    def run_one(self):
        qjobs = QueuedJob.objects.all().filter(q=self.q, done=False).order_by('priority', 'enqueuetime')
        if len(qjobs) == 0:
            print 'No jobs; sleeping.'
            return False

        qjob = None
        for j in qjobs:
            # HACK - really I want to check that I have an index that
            # another worker hasn't already applied to this job.
            if j.work.all().filter(worker=self.worker).count():
                # I've already worked on this one.
                continue
            else:
                qjob = j
                break

        if qjob is None:
            print "No jobs (that I haven't already worked on); sleeping."
            return False
            
        self.worker.job = qjob
        self.worker.save()

        print 'Working on job', qjob
        # FIXME
        #w = Work(job=qjob, worker=self.worker, inprogress=True)
        #w.save()

        # retrieve the input files.
        url = qjob.get_url()
        print 'Retrieving URL %s...' % url

        job = qjob.job

        (f, axy) = tempfile.mkstemp('', 'axy-%s-' % job.jobid)
        os.close(f)
        
        fn = qjob.retrieve_to_file(axy)
        print 'Saved as', fn

        tmpdir = tempfile.mkdtemp('', 'backend-results-')
        tarfile = os.path.join(tmpdir, 'results.tar')
        backend = 'backend2'
        cancelfile = os.path.join(tmpdir, 'cancel')
        # HACK - pipes?
        cmd = ('cd %(tempdir)s; %(backend)s -c %(backendconf)s -C %(cancel)s -v %(axy)s > %(logfile)s 2>&1' %
               dict(tempdir=tmpdir, backend=backend,
                    backendconf=self.backendcfg,
                    cancel=cancelfile,
                    axy=axy,
                    logfile='blind.log'))
        print 'Running command', cmd
    
        (rtn, out, err) = run_command(cmd, timeout=1,
                                      callback=lambda: self.callback(qjob, cancelfile))

        if rtn:
            print 'backend failed: rtn val %i' % rtn, ', out', out, ', err', err

        # Send results -- only if solved??
        solvedfile = os.path.join(tmpdir, 'solved')
        if os.path.exists(solvedfile):
            cmd = 'cd %s; tar cf %s *' % (tmpdir, tarfile)
            (rtn, out, err) = run_command(cmd)
            if rtn:
                print 'tar failed: rtn val %i' % rtn, ', out', out, ', err', err
            url = qjob.get_put_results_url()
            f = open(tarfile, 'rb')
            tardata = f.read()
            f.close()
            print 'Tardata string is %i bytes long.' % len(tardata)
            tardata = tardata.encode('base64_codec')
            print 'Encoded string is %i bytes long.' % len(tardata)
            data = urlencode({ 'tar': tardata })
            print 'Sending response to', url
            print 'url-encoded string is %i bytes long.' % len(data)
            f = urlopen(url, data)
            response = f.read()
            f.close()
            print 'Got response:', response

            job.set_status('Solved')

            # Add WCS to database.
            wcsfile = os.path.join(tmpdir, 'wcs.fits')
            wcs = TanWCS(file=wcsfile)
            wcs.save()

            # HACK - need to make blind write out raw TAN, tweaked TAN, and tweaked SIP.
            # HACK - compute ramin, ramax, decmin, decmax.
            calib = Calibration(raw_tan = wcs)
            calib.save()
            job.calibration = calib
            job.add_machine_tags()
        else:
            job.set_status('Failed', 'Did not solve.')

        job.save()

        qjob.inprogress = False
        qjob.done = True
        qjob.save()

        # FIXME
        #w.inprogress = False
        #w.done = True
        #w.save()
        self.worker.job = None
        self.worker.save()

        # HACK - delete tempfiles.
        return True


    def callback(self, job, fn):
        js=QueuedJob.objects.all().filter(job=job)
        if js.count() == 0:
            return
        j=js[0]
        if j.done:
            print 'Touching file', fn
            f = open(fn, 'wb')
            f.write('')
            f.close()

    def run(self):
        while True:
            if not self.run_one():
                time.sleep(5)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        indexdirs = sys.argv[1:]
    else:
        indexdirs = [
            '/home/gmaps/INDEXES/500',
            ]

    (q,nil) = JobQueue.objects.get_or_create(name=settings.SITE_ID, queuetype='solve')
    s = Solver(q, indexdirs)
    s.run()

