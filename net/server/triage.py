import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import astrometry.net.settings as settings

os.environ['LD_LIBRARY_PATH'] = settings.UTIL_DIR
os.environ['PATH'] = ':'.join(['/bin', '/usr/bin',
                               settings.BLIND_DIR,
                               settings.UTIL_DIR,
                               ])
# This must match the Apache setting UPLOAD_DIR
os.environ['UPLOAD_DIR'] = settings.UPLOAD_DIR

import sys
import socket
import time
import tempfile
import thread
from datetime import datetime
from math import sqrt

from urllib import urlencode, urlretrieve
from urllib2 import urlopen
from urlparse import urlparse

import pyfits

from django.db.models import Q

settings.SERVER_LOGFILE = 'worker.log'

from astrometry.net.server.log import log
from astrometry.net.server.models import *

from astrometry.net.portal.job import Job, Submission, DiskFile, Calibration, Tag
from astrometry.net.upload.models import UploadedFile
from astrometry.net.portal.convert import convert, is_tarball, get_objs_in_field, FileConversionError
from astrometry.net.portal.wcs import TanWCS
from astrometry.net.server import indexes

from astrometry.util.run_command import run_command
import astrometry.util.sip as sip

blindlog = 'blind.log'

def bailout(job, reason):
    job.set_status('Failed', reason)
    job.save()

def file_get_contents(fn):
    f = open(fn, 'rb')
    txt = f.read()
    f.close()
    return txt

def userlog(*msg):
    f = open(blindlog, 'a')
    f.write(' '.join(map(str, msg)) + '\n')
    f.close()

class Triage(object):
    def __init__(self, q):
        self.q = q
        self.worker = Worker(queue=q)
        self.worker.save()
        self.worker.start_keepalive_thread()

    def run_one(self):
        qjobs = QueuedJob.objects.all().filter(q=self.q, done=False, inprogress=False).order_by('priority', 'enqueuetime')
        if len(qjobs) == 0:
            print 'No jobs; sleeping.'
            return False
        qjob = qjobs[0]

        qjob.inprogress = True
        qjob.save()

        if qjob.job:
            self.run_job(qjob)
        elif qjob.submission:
            self.run_submission(qjob)

        return True

    def run_submission(self, qjob):
        submission = qjob.submission
        log('Running submission:')
        log(str(submission))
        tmpfile = None
        basename = None

        if submission.datasrc == 'url':
            # download the URL to a new temp file.
            userlog('Retrieving URL...')
            (fd, tmpfile) = tempfile.mkstemp('', 'download')
            os.close(fd)
            log('Retrieving URL ' + submission.url + ' to file ' + tmpfile)
            f = urlretrieve(submission.url, tmpfile)
            p = urlparse(submission.url)
            p = p[2]
            if p:
                s = p.split('/')
                basename = s[-1]

        elif submission.datasrc == 'file':
            tmpfile = submission.uploaded.get_filename()
            log('uploaded tempfile is ' + tmpfile)
            basename = submission.uploaded.userfilename

        else:
            bailout(job, 'no datasrc')
            return False

        df = DiskFile.for_file(tmpfile)
        df.save()
        submission.diskfile = df
        submission.fileorigname = basename
        submission.save()

        # Handle compressed files.
        uncomp = convert(submission, submission.diskfile, 'uncomp-js')
        log('uncompressed file: %s' % uncomp)

        # Handle tar files: add a Submission, create new Jobs.
        job = None

        if is_tarball(uncomp):
            log('file is tarball.')
            # create temp dir to extract tarfile.
            tempdir = tempfile.mkdtemp('', 'tarball-')
            cmd = 'tar xvf %s -C %s' % (uncomp, tempdir)
            userlog('Extracting tarball...')
            (rtn, out, err) = run_command(cmd)
            if rtn:
                userlog('Failed to un-tar file:\n' + err)
                bailout(submission, 'failed to extract tar file')
                return False
            fns = out.strip('\n').split('\n')
            self.run_tarball(tempdir, fns, submission)
            shutil.rmtree(tempdir)
            return True

        else:
            # Not a tarball.
            job = Job(
                submission = submission,
                jobid = submission.subid,
                fileorigname = submission.fileorigname,
                diskfile = submission.diskfile,
                )
            job.enqueuetime = submission.submittime
            job.set_is_duplicate()
            job.save()
            submission.save()

            qjob.submission = None
            qjob.job = job
            qjob.save()

            return self.run_job(qjob)

    def run_tarball(self, basedir, filenames, submission):
        validpaths = []
        for fn in filenames:
            path = os.path.join(basedir, fn)
            log('Path "%s"' % path)
            if not os.path.exists(path):
                log('Path "%s" does not exist.' % path)
                continue
            if os.path.islink(path):
                log('Path "%s" is a symlink.' % path)
                continue
            if os.path.isfile(path):
                validpaths.append(path)
            else:
                log('Path "%s" is not a file.' % path)

        if len(validpaths) == 0:
            userlog('Tar file contains no regular files.')
            bailout(submission, "tar file contains no regular files.")
            return -1

        log('Got %i paths.' % len(validpaths))

        for p in validpaths:
            origname = os.path.basename(p)
            df = DiskFile.for_file(p)
            df.save()
            log('New diskfile ' + str(df.filehash))

            if len(validpaths) == 1:
                jobid = submission.subid
            else:
                jobid = Job.generate_jobid()

            job = Job(
                jobid = jobid,
                submission = submission,
                diskfile = df,
                fileorigname = origname,
                )

            if len(validpaths) == 1:
                job.enqueuetime = submission.submittime
                job.set_is_duplicate()
                job.save()
                submission.save()
                # One file in tarball: convert straight to a Job.
                log('Single-file tarball.')
                rtn = handle_job(job, sshconfig)
                if rtn:
                    return rtn
                break

            job.set_enqueuetime_now()
            job.set_is_duplicate()
            job.set_status('Queued')
            job.save()
            submission.save()
            log('Enqueuing Job: ' + str(job))
            Job.submit_job_or_submission(job)


    def run_job(self, qjob):
        me = self.worker

        job = qjob.job
        log('handle_job: queued job', qjob)
        log('handle_job: job', job)

        me.job = qjob
        me.save()

        job.set_status('Running')
        job.save()

        df = job.diskfile
        log('diskfile is %s' % df.get_path())

        submission = job.submission
        jobid = job.jobid

        # create and go to the job directory.
        os.umask(07)
        job.create_job_dir()
        jobdir = job.get_job_dir()
        os.chdir(jobdir)

        # Handle compressed files.
        uncomp = convert(job, df, 'uncomp')
        log('uncompressed file is %s' % uncomp)

        # Compute hash of uncompressed file.
        # field.compute_filehash(uncomp)

        axy = 'job.axy'
        axypath = job.get_filename(axy)
        axyargs = {}
        filetype = job.submission.filetype

        if filetype == 'image':

            # FIXME - chuck this in another queue for source extraction...

            log('source extraction...')
            userlog('Doing source extraction...')

            convert(job, df, 'getimagesize')
            if (df.imagew * df.imageh) > 5000000:  # 5 MPixels
                userlog('Downsampling your image...')
                target = 'xyls-half-sorted'
            else:
                target = 'xyls-sorted'

            try:
                log('image2xy...')
                xylist = convert(job, df, target)
                log('xylist is', xylist)
            except FileConversionError,e:
                userlog('Source extraction failed.')
                bailout(job, 'Source extraction failed.')
                return False
            log('created xylist %s' % xylist)

            axyargs['-x'] = xylist

        elif (filetype == 'fits') or (filetype == 'text'):
            if filetype == 'text':
                df.filetype = 'text'
                try:
                    userlog('Parsing your text file...')
                    xylist = convert(job, df, 'xyls')
                    log('xylist is', xylist)
                except FileConversionError,e:
                    userlog('Parsing your text file failed.')
                    bailout(job, 'Parsing text file failed.')
                    return False

            else:
                df.filetype = 'xyls'
                try:
                    log('fits2fits...')
                    xylist = convert(job, df, 'xyls')
                    log('xylist is', xylist)
                except FileConversionError,e:
                    userlog('Sanitizing your FITS file failed.')
                    bailout(job, 'Sanitizing FITS file failed.')
                    return False

                (xcol, ycol) = job.get_xy_cols()
                if xcol:
                    axyargs['--x-column'] = xcol
                    if ycol:
                        axyargs['--y-column'] = ycol

                log('created xylist %s' % xylist)
                    
            cmd = 'xylsinfo %s' % xylist
            (rtn, out, err) = run_command(cmd)
            if rtn:
                log('out: ' + out)
                log('err: ' + err)
                bailout(job, 'Getting xylist image size failed: ' + err)
                return False
            lines = out.strip().split('\n')
            info = {}
            for l in lines:
                t = l.split(' ')
                info[t[0]] = t[1]
            if 'imagew' in info:
                width = float(info['imagew'])
            else:
                width = float(info['width'])
            if 'imageh' in info:
                height = float(info['imageh'])
            else:
                height = float(info['height'])

            axyargs.update({
                '-x': xylist,
                '--width' : width,
                '--height' : height,
                '--no-fits2fits' : None,
                })
            df.imagew = width
            df.imageh = height

        else:
            bailout(job, 'no filetype')
            return False

        rtnval = os.fork()
        if rtnval == 0:
            # I'm the child process
            rtnval = os.EX_OK
            try:
                produce_alternate_xylists(job)
            except:
                log('Something bad happened at produce_alternate_xylists.')
                rtnval = os.EX_SOFTWARE
                os._exit(rtnval)
            else:
                # I'm the parent - carry on.
                pass

        (lower, upper) = job.get_scale_bounds()
        units = job.get_scaleunits()

        axyargs.update({
            '-o' : axy,
            '--scale-units' : units,
            '--scale-low' : lower,
            '--scale-high' : upper,
            '--fields' : 1,
            '--wcs' : 'wcs.fits',
            '--rdls' : 'index.rd.fits',
            '--match' : 'match.fits',
            '--solved' : 'solved',
            })

        (dotweak, tweakorder) = job.get_tweak()
        log('do tweak?', dotweak, 'order', tweakorder)
        if dotweak:
            axyargs['--tweak-order'] = tweakorder
        else:
            axyargs['--no-tweak'] = None

        cmd = 'augment-xylist ' + ' '.join(k + ((v and ' ' + str(v)) or '') for (k,v) in axyargs.items())
            
        log('running: ' + cmd)
        (rtn, out, err) = run_command(cmd)
        if rtn:
            log('out: ' + out)
            log('err: ' + err)
            bailout(job, 'Creating axy file failed: ' + err)
            return False

        log('created axy file ' + axypath)

        df.save()

        # Decide which indexes should be applied, and create Work entries
        # for each of them.
        # read the axy we just wrote...
        hdus = pyfits.open(axypath)
        hdr = hdus[0].header
        scalelo = float(hdr['ANAPPL1'])
        scalehi = float(hdr['ANAPPU1'])
        imagew  = float(hdr['IMAGEW' ])
        imageh  = float(hdr['IMAGEH' ])

        minsize = scalelo * min(imagew, imageh)
        maxsize = scalehi * sqrt(imagew**2 + imageh**2)
        # MAGIC - minimum size of a quad in the field.
        minsize *= 0.1

        minsize = arcsec2rad(minsize)
        maxsize = arcsec2rad(maxsize)

        inds = Index.objects.all().filter(
            scalelo__lte=maxsize, scalehi__gte=minsize)
        # Re-enqueue in the solving queue.

        (q,nil) = JobQueue.objects.get_or_create(name=qjob.q.name,
                                                 queuetype='solve')
        qnew = QueuedJob(q=q, enqueuetime=Job.timenow(), job=job)
        qnew.save()
        log('Enqueuing in "%s", Job "%s"' % (q, qnew))

        for index in inds:
            w = Work(job=qnew, index=index)
            w.save()

        print 'Added work: ', ', '.join([str(w.index) for w in qnew.work.all()])

        qjob.inprogress = False
        qjob.done = True
        qjob.save()

        self.worker.job = None
        self.worker.save()
        
        return True

    def run(self):
        while True:
            if not self.run_one():
                time.sleep(5)


if __name__ == '__main__':

    indexes.load_indexes()

    (q,nil) = JobQueue.objects.get_or_create(name=settings.SITE_NAME, queuetype='triage')
    t = Triage(q)
    t.run()

