#! /usr/bin/env python

import os
import sys
import tempfile
import traceback

from urlparse import urlparse

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
sys.path.extend(['/home/gmaps/test',
                 '/home/gmaps/django/lib/python2.4/site-packages'])

import astrometry.net.settings as settings

os.environ['LD_LIBRARY_PATH'] = settings.UTIL_DIR
os.environ['PATH'] = ':'.join(['/bin', '/usr/bin',
                               settings.BLIND_DIR,
                               settings.UTIL_DIR,
                               ])
# This must match the Apache setting UPLOAD_DIR
os.environ['UPLOAD_DIR'] = settings.UPLOAD_DIR

import logging
import os.path
import urllib
import shutil

from django.db import models

from astrometry.net.portal.models import Job, Submission, DiskFile, Calibration, Tag
from astrometry.net.upload.models import UploadedFile
from astrometry.net.portal.log import log
from astrometry.net.portal.convert import convert, is_tarball, get_objs_in_field, FileConversionError
from astrometry.net.portal.wcs import TanWCS
from astrometry.util.run_command import run_command

import astrometry.util.sip as sip

def bailout(job, reason):
    job.set_status('Failed', reason)
    job.save()

blindlog = 'blind.log'

def file_get_contents(fn):
    f = open(fn, 'r')
    txt = f.read()
    f.close()
    return txt

def userlog(*msg):
    f = open(blindlog, 'a')
    f.write(' '.join(map(str, msg)) + '\n')
    f.close()

# Returns True if the job was processed successfully (ie it finished)
def handle_job(job, sshconfig):
    try:
        return real_handle_job(job, sshconfig)
    except Exception,e:
        errstr = str(e)
        log('Failed with exception: ', str(e))
        log('--------------')
        log(traceback.format_exc())
        log('--------------')
        job.set_status('Failed', errstr)
        job.save()
    return False

def produce_alternate_xylists(job):
    log("I'm producing alternate xylists like nobody's bidness.")
    df = job.diskfile

    for n in [1, 2, 3, 4]:
        log('Producing xyls variant %i...' % n)
        convert(job, df, 'xyls', { 'variant': n })

# Returns True if the job was processed successfully (ie it finished)
def real_handle_job(job, sshconfig):
    log('handle_job: ' + str(job))

    job.set_status('Running')
    job.save()

    df = job.diskfile
    log('diskfile is %s' % df.get_path())

    submission = job.submission
    jobid = job.jobid

    # go to the job directory.
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

    # shell into compute server...

    # For the "test" instance:
    #  ssh an_remote_test  (as gmaps)
    #  runs cluster24:amd-an-2/quads/an-remote-test.sh
    #  run backend with amd-an-2/quads/backend-test.cfg

    cmd = ('(echo %(jobid)s; '
           ' tar cf - --ignore-failed-read %(axyfile)s) | '
           'ssh -x -T %(sshconfig)s 2>>%(logfile)s | '
           'tar xf - --atime-preserve -m --exclude=%(axyfile)s '
           '>>%(logfile)s 2>&1' %
           dict(jobid=jobid, axyfile=axy,
                sshconfig=sshconfig, logfile=blindlog)
           + '; chmod 664 *; chgrp www-data *')
    # --group G --mode M --owner O ?

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

def handle_tarball(basedir, filenames, submission):
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

def main(sshconfig, joblink):
    if not os.path.islink(joblink):
        log('Expected second argument to be a symlink; "%s" isn\'t.' % joblink)
        return -1

    # jobfile should be a symlink; get the destination.
    jobdir = os.readlink(joblink)

    # go to the job directory.
    os.chdir(jobdir)

    # the name of the symlink is the jobid.
    jobid = os.path.basename(joblink)

    # if it's a Job...
    jobs = Job.objects.all().filter(jobid = jobid)
    if len(jobs):
        return handle_job(jobs[0], sshconfig)

    # else it's a Submission...
    submissions = Submission.objects.all().filter(subid=jobid)
    if len(submissions) != 1:
        log('Found %i submissions, not 1' % len(submissions))
        return False
    submission = submissions[0]
    log('Running submission: ' + str(submission))

    tmpfile = None
    basename = None

    if submission.datasrc == 'url':
        # download the URL to a new temp file.
        userlog('Retrieving URL...')
        (fd, tmpfile) = tempfile.mkstemp('', 'download')
        os.close(fd)
        log('Retrieving URL ' + submission.url + ' to file ' + tmpfile)
        f = urllib.urlretrieve(submission.url, tmpfile)
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

    # Handle tar files: add a Submission, create new Jobs.
    job = None

    log('uncompressed file: %s' % uncomp)

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
        handle_tarball(tempdir, fns, submission)
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
        return handle_job(job, sshconfig)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: %s <ssh-config> <input-file>' % sys.argv[0]
        sys.exit(-1)
    sshconfig = sys.argv[1]
    joblink = sys.argv[2]

    os.umask(07)

    ok = main(sshconfig, joblink)

    if ok:
        # remove the symlink to indicate that we've successfully finished this
        # job.
        os.unlink(joblink)
        sys.exit(0)
    sys.exit(-1)
