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

from astrometry.net.portal.models import Job, Submission, DiskFile, Calibration, Tag
from astrometry.net.upload.models import UploadedFile
from astrometry.net.portal.convert import convert, is_tarball, get_objs_in_field, FileConversionError
from astrometry.net.portal.wcs import TanWCS

from astrometry.net.util.run_command import run_command
import astrometry.util.sip as sip

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


# QueuedJob qjob, Worker me
def real_handle_job(qjob, me):
    qjob.inprogress = True
    qjob.save()

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
    #field.compute_filehash(uncomp)

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
        #log('out: ', out)
        #log('lines: ', str(lines))
        info = {}
        for l in lines:
            t = l.split(' ')
            info[t[0]] = t[1]
        #log('info: ', str(info))
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

    # Re-enqueue in the solving queue.

    q = JobQueue.objects.get_or_create(name=qjob.q.name,
                                       queuetype='solve')

    qnew = QueuedJob(q=q, enqueuetime=Job.timenow(), job=job)
    qnew.save()
    #qjob.q = q
    #qjob.inprogress = False
    #qjob.enqueuetime=Job.timenow()
    #qjob.save()

    return True


    submitpath = '/home/gmaps/test/astrometry/server/submit.py'

    jobqueue = settings.SITE_ID

    cmd = ('%(submit)s %(jobqueue)s %(jobid)s %(axyfile)s'
           ' >> %(logfile)s 2>&1' %
           dict(submit=submitpath,
                jobqueue=jobqueue,
                jobid=jobid,
                axyfile=axy,
                logfile=blindlog))
                #logfile=settings.PORTAL_LOGFILE))


    job.set_status('Running')
    job.set_starttime_now()
    job.save()

    log('Running command:', cmd)
    w = os.system(cmd)

    job.set_finishtime_now()
    job.save()


    if not os.WIFEXITED(w):
        bailout(job, 'Solver didn\'t exit normally.')
        return False

    rtn = os.WEXITSTATUS(w)
    if rtn:
        log('Solver failed with return value %i' % rtn)
        bailout(job, 'Solver failed.')
        return False

    log('Command completed successfully.')

    # Record results in the job database.

    if os.path.exists(job.get_filename('solved')):
        job.set_status('Solved')

        # Add WCS to database.
        wcs = TanWCS(file=job.get_filename('wcs.fits'))
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
    return True


