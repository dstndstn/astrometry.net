#! /usr/bin/env python

import astrometry.net.django_commandline

import os
import sys
import tempfile
import traceback

import logging
import os.path
import urllib
import urllib2
import shutil
import tarfile
import re

from urlparse import urlparse
from urllib import urlencode
from urllib2 import urlopen
from StringIO import StringIO

from django.db import models

from astrometry.net.portal.models import Job, Submission, DiskFile, Calibration, Tag
from astrometry.net.upload.models import UploadedFile
from astrometry.net.portal.log import log
from astrometry.net.portal.convert import convert, is_tarball, FileConversionError
from astrometry.net.portal.wcs import TanWCS
from astrometry.util.run_command import run_command
from astrometry.util.file import *
from astrometry.util.shell import *

from astrometry.net.server import ssh_master

import astrometry.util.sip as sip

blindlog = 'blind.log'

def is_youtube_vid(url):
    # http://www.youtube.com/watch?v=wyJYPIWF3-0
    regex = re.compile(r'http://(www.)?youtube\.com/watch\?v=(.*)')
    match = regex.match(url)
    if match is None:
        return False
    log('YouTube regex matched: ', url, ' -- vid id', match.group(2))
    return True

class Watcher(object):
    def __init__(self):
        self.bailedout = False
        
    def bailout(self, job, reason):
        self.bailedout = True
        log('Bailing out:', reason)
        job.set_error_status(reason)
        job.save()

    def userlog(self, *msg):
        f = open(blindlog, 'a')
        f.write(' '.join(map(str, msg)) + '\n')
        f.close()

    # Returns True if the job was processed successfully (ie it finished)
    def handle_job(self, job):
        try:
            return self.real_handle_job(job)
        except Exception,e:
            errstr = str(e)
            log('Failed with exception: ', str(e))
            log('--------------')
            log(traceback.format_exc())
            log('--------------')
            job.set_error_status(errstr)
            job.save()
        return False

    def produce_alternate_xylists(self, job):
        log("I'm producing alternate xylists like nobody's bidness.")
        for n in [1, 2, 3, 4]:
            log('Producing xyls variant %i...' % n)
            convert(job, 'xyls', { 'variant': n })

    # Returns True if the job was processed successfully (ie it finished)
    def real_handle_job(self, job):
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
        uncomp = convert(job, 'uncomp')
        log('uncompressed file is %s' % uncomp)

        # Compute hash of uncompressed file.
        #field.compute_filehash(uncomp)

        axypath = job.get_axy_filename()
        axyargs = {}

        filetype = job.submission.filetype

        if filetype == 'image':
            log('source extraction...')
            self.userlog('Doing source extraction...')
            convert(job, 'getimagesize')
            if (df.imagew * df.imageh) > 5000000:  # 5 MPixels
                self.userlog('Downsampling your image...')
                target = 'xyls-half-sorted'
            else:
                target = 'xyls-sorted'

            try:
                log('image2xy...')
                xylist = convert(job, target)
                log('xylist is', xylist)
            except FileConversionError,e:
                self.userlog('Source extraction failed.')
                self.bailout(job, 'Source extraction failed.')
                return False
            log('created xylist %s' % xylist)
            axyargs['-x'] = xylist

        elif (filetype == 'fits') or (filetype == 'text'):
            if filetype == 'text':
                df.filetype = 'text'
                try:
                    self.userlog('Parsing your text file...')
                    xylist = convert(job, 'xyls')
                    log('xylist is', xylist)
                except FileConversionError,e:
                    self.userlog('Parsing your text file failed.')
                    self.bailout(job, 'Parsing text file failed.')
                    return False
            else:
                df.filetype = 'xyls'
                try:
                    log('fits2fits...')
                    xylist = convert(job, 'xyls')
                    log('xylist is', xylist)
                except FileConversionError,e:
                    self.userlog('Sanitizing your FITS file failed.')
                    self.bailout(job, 'Sanitizing FITS file failed.')
                    return False

                (xcol, ycol) = job.get_xy_cols()
                if xcol:
                    axyargs['--x-column'] = xcol
                if ycol:
                    axyargs['--y-column'] = ycol

            log('created xylist %s' % xylist)
            cmd = 'xylsinfo'
            if xcol:
                cmd += ' -X %s' % xcol
            if ycol:
                cmd += ' -Y %s' % ycol
            cmd += ' %s' % xylist
            (rtn, out, err) = run_command(cmd)
            if rtn:
                log('command: ' + cmd)
                log('out: ' + out)
                log('err: ' + err)
                self.bailout(job, 'Getting xylist image size failed: ' + err)
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
            df.imagew = int(round(width ))
            df.imageh = int(round(height))
            df.save()

        else:
            self.bailout(job, 'no filetype')
            return False

        if False:
            rtnval = os.fork()
            if rtnval == 0:
                # I'm the child process
                rtnval = os.EX_OK
                try:
                    self.produce_alternate_xylists(job)
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
            '-o' : axypath,
            '--scale-units' : units,
            '--scale-low' : lower,
            '--scale-high' : upper,
            '--fields' : 1,
            '--wcs' : 'wcs.fits',
            '--rdls' : 'index.rd.fits',
            '--match' : 'match.fits',
            '--solved' : '../solved',
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
            self.bailout(job, 'Creating axy file failed: ' + err)
            return False

        log('created axy file ' + axypath)
        df.save()
        job.set_status('Running')
        job.set_starttime_now()
        job.save()

        tardata = self.solve_job(job)

        if tardata is not None:
            # extract the resulting tarball...
            f = StringIO(tardata)
            tar = tarfile.open(name='', mode='r|', fileobj=f)
            for tarinfo in tar:
                log('  ', tarinfo.name, 'is', tarinfo.size, 'bytes in size')
                tar.extract(tarinfo, job.get_job_dir())
            tar.close()
            f.close()
            # chmod 664 *; chgrp www-data *

        job.set_finishtime_now()
        job.save()

        # Record results in the job database.
        if os.path.exists(job.get_filename('wcs.fits')):
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
            job.set_status('Unsolved')

        job.save()
        return True

    def handle_tarball(self, basedir, filenames, submission):
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
            self.userlog('Tar file contains no regular files.')
            self.bailout(submission, "tar file contains no regular files.")
            return -1

        log('Got %i paths.' % len(validpaths))

        if len(validpaths) > 1:
            submission.multijob = True
            submission.save()

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
                submission.alljobsadded = True
                submission.save()
                # One file in tarball: convert straight to a Job.
                log('Single-file tarball.')
                rtn = self.handle_job(job)
                if rtn:
                    return rtn
                break

            job.set_enqueuetime_now()
            job.set_is_duplicate()
            job.set_status('Queued')
            job.save()
            log('Enqueuing Job: ' + str(job))
            Job.submit_job_or_submission(job)
            submission.alljobsadded = True
            submission.save()

    def download_url(self, submission, tmpfile):
        self.userlog('Retrieving URL...')
        log('Retrieving URL ' + submission.url + ' to file ' + tmpfile)

        if is_youtube_vid(submission.url):
            self.handle_youtube_vid(submission, tmpfile)
            return None
        
        # download the URL to a new temp file.
        try:
            f = urlopen(submission.url)
            fout = open(tmpfile, 'wb')
            fout.write(f.read())
            fout.close()
            f.close()
        except urllib2.HTTPError, e:
            self.bailout(submission, 'Failed to retrieve URL: ' + str(e))
            return False
        log('URL info for %s:' % f.geturl())
        for k,v in f.info().items():
            log('  ',k,'=',v)
        p = urlparse(submission.url)
        p = p[2]
        if p:
            s = p.split('/')
            basename = s[-1]
        return basename

    def handle_youtube_vid(self, submission, tmpfile):
        self.userlog('Youtube video.')
        cmd = 'youtube-dl -o %s \"%s\"' % (tmpfile, shell_escape_inside_quotes(submission.url))
        log('running command: %s' % cmd)
        (rtn, out, err) = run_command(cmd)
        if rtn:
            log('out: ' + out)
            log('err: ' + err)
            self.bailout(submission, 'Downloading youtube video failed: ' + err)
            return None
        # create temp dir to extract tarfile.
        tempdir = tempfile.mkdtemp('', 'videoframes-')
        log('writing frames to tempdir %s' % tempdir)
        cmd = 'mplayer -vo pnm:outdir=%s %s' % (tempdir, tmpfile)
        log('running command: %s' % cmd)
        (rtn, out, err) = run_command(cmd)
        if rtn:
            log('out: ' + out)
            log('err: ' + err)
            self.bailout(submission, 'Extracting youtube video failed: ' + err)
            return None
        fns = os.listdir(tempdir)
        # yoink
        self.handle_tarball(tempdir, fns, submission)
        shutil.rmtree(tempdir)
        return None
    
    def run_link(self, joblink):
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
            return self.handle_job(jobs[0])
        # else it's a Submission...
        submissions = Submission.objects.all().filter(subid=jobid)
        if len(submissions) != 1:
            log('Found %i submissions, not 1' % len(submissions))
            return False
        submission = submissions[0]
        log('Running submission: ' + str(submission))
        return self.run_sub(None, submission)

    def run_sub(self, submission):
        tmpfile = None
        basename = None

        if submission.datasrc == 'url':
            (fd, tmpfile) = tempfile.mkstemp('', 'download-')
            os.close(fd)
            basename = self.download_url(submission, tmpfile)
            if self.bailedout:
                return False
            if basename is None:
                return True

        elif submission.datasrc == 'file':
            tmpfile = submission.uploaded.get_filename()
            log('uploaded tempfile is ' + tmpfile)
            basename = submission.uploaded.userfilename

        else:
            self.bailout(job, 'no datasrc')
            return False

        df = DiskFile.for_file(tmpfile)
        df.save()
        submission.diskfile = df
        submission.fileorigname = basename
        submission.save()
        # Handle compressed files.
        uncomp = convert(submission, 'uncomp-js')
        # Handle tar files: add a Submission, create new Jobs.
        job = None
        log('uncompressed file: %s' % uncomp)
        if is_tarball(uncomp):
            log('file is tarball.')
            # create temp dir to extract tarfile.
            tempdir = tempfile.mkdtemp('', 'tarball-')
            cmd = 'tar xvf %s -C %s' % (uncomp, tempdir)
            self.userlog('Extracting tarball...')
            (rtn, out, err) = run_command(cmd)
            if rtn:
                self.userlog('Failed to un-tar file:\n' + err)
                self.bailout(submission, 'failed to extract tar file')
                return False
            fns = out.strip('\n').split('\n')
            self.handle_tarball(tempdir, fns, submission)
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
            submission.alljobsadded = True
            submission.save()
            return self.handle_job(job)


    def main(self, joblink):
        os.umask(07)
        ok = self.run_link(joblink)
        if ok:
            # remove the symlink to indicate that we've successfully finished this
            # job.
            os.unlink(joblink)
            return 0
        return -1

    def solve_job(self, job):
        return None


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s <input-file>' % sys.argv[0]
        sys.exit(-1)
    joblink = sys.argv[1]
    w = Watcher()
    sys.exit(w.main(joblink))
