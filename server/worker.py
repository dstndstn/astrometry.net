import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.server.settings'

import socket
import time
import tempfile

from urllib import urlencode
from urllib2 import urlopen

import pyfits

import astrometry.server.settings as settings

settings.LOGFILE = 'worker.log'

from astrometry.server.log import log
from astrometry.server.models import *

from astrometry.net.util.run_command import run_command

def get_header(header, key, default):
    #if key in header:
    #    return header[key]
    #return default
    try:
        return header[key]
    except KeyError:
        return default

def callback(jobid, fn):
    print 'callback.'
    js=QueuedJob.objects.all().filter(jobid=jobid)
    if js.count() == 0:
        return
    j=js[0]
    if j.stopwork:
        print 'Touching file', fn
        f = open(fn, 'wb')
        f.write('')
        f.close()

def main():
    qname = 'test'
    #backendconfig = 'backend.cfg'
    indexdirs = [
        '/home/gmaps/INDEXES/500',
        'INDEXES',
        ]

    q = JobQueue.objects.get(name=qname)

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    me = Worker(hostname=hostname,
                ip=ip)
    me.save()

    #f = open(backendconfig, 'rb')
    #be = f.read()
    #f.close()
    #lines = be.split('\n')

    indexes = []
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
                #print 'file', f, 'base', base
                if (os.path.exists(base + qsuff) and
                    os.path.exists(base + ssuff)):
                    print 'Found index', base
                    qfile = base + qsuff
                    hdus = pyfits.open(qfile)
                    hdr = hdus[0].header
                    indexid = get_header(hdr, 'INDEXID', None)
                    hp = get_header(hdr, 'HEALPIX', -1)
                    hpnside = get_header(hdr, 'HPNSIDE', 1)
                    print 'id', indexid
                    print 'hp', hp
                    print 'hp nside', hpnside
                    if indexid is not None:
                        indexes.append((base, indexid, hp, hpnside))

    for (fn, indexid, hp, hpnside) in indexes:
        print 'Saving index', fn
        li = Index(indexid=indexid,
                   healpix=hp,
                   healpix_nside=hpnside,
                   worker=me)
        li.save()

    print 'My indexes:', me.pretty_index_list()

    (f, backendcfg) = tempfile.mkstemp('', 'backend.cfg-')
    os.close(f)
    f = open(backendcfg, 'wb')
    f.write('\n'.join(['inparallel'] +
                      ['index %s' % path for (path, indid, hp, hpnside) in indexes]
                      ))
    f.close()

    while True:
        jobs = QueuedJob.objects.all().filter(q=q, stopwork=False).order_by('priority', 'enqueuetime')
        if len(jobs) == 0:
            print 'No jobs; sleeping.'
            time.sleep(5)
            continue

        job = None
        for j in jobs:

            # HACK - really I want to check that I have an index that
            # another work hasn't already applied to this job.
            if j.work.all().filter(worker=me).count():
                # I've already worked on this one.
                continue
            else:
                job = j
                break

        if job is None:
            print "No jobs (that I haven't already worked on); sleeping."
            time.sleep(5)
            continue
            
        me.job = job
        me.save()
        print 'Working on job', job
        w = Work(job=job, worker=me, inprogress=True)
        w.save()

        # retrieve the input files.
        url = job.get_url()
        print 'Retrieving URL %s...' % url

        (f, axy) = tempfile.mkstemp('', 'axy-%s-' % job.jobid)
        os.close(f)
        
        fn = job.retrieve_to_file(axy)
        print 'Saved as', fn

        tmpdir = tempfile.mkdtemp('', 'backend-results-')

        tarfile = os.path.join(tmpdir, 'results.tar')

        # HACK
        backend = '/home/gmaps/test/astrometry/blind/backend'
        # HACK - pipes?
        cmd = 'cd %s; %s -c %s %s; tar cf %s *' % (tmpdir, backend, backendcfg, axy, tarfile)
        print 'Running command', cmd

        cancelfile = '/tmp/cancel'
        run_command(cmd, timeout=1,
                    callback=lambda: callback(job.jobid, cancelfile))

        # Send results -- only if solved??
        solvedfile = os.path.join(tmpdir, 'solved')
        if os.path.exists(solvedfile):
            url = job.get_put_results_url()
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

            job.stopwork = True
            job.save()

        w.inprogress = False
        w.done = True
        w.save()
        me.job = None
        me.save()

        # HACK - delete tempfiles.


if __name__ == '__main__':
    main()

