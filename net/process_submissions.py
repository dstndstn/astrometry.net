#! /usr/bin/env python

import os
import sys

# add .. to PYTHONPATH
path = os.path.abspath(__file__)
basedir = os.path.dirname(os.path.dirname(path))
sys.path.append(basedir)

# add ../blind and ../util to PATH
os.environ['PATH'] += ':' + os.path.join(basedir, 'blind')
os.environ['PATH'] += ':' + os.path.join(basedir, 'util')

import tempfile
import traceback
from urlparse import urlparse
import logging
import urllib
import shutil
import multiprocessing
import time
import re

import logging
#logging.basicConfig(format='%(message)s',
#                    level=logging.DEBUG)

from astrometry.util import image2pnm
from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command

from astrometry.util.util import Tan

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
#import astrometry.net.settings as settings
import settings
settings.LOGGING['loggers'][''] = {
    'handlers': ['console'],
    'level': 'DEBUG',
    'propagate': True,
}
from astrometry.net.models import *
from log import *

from django.utils.log import dictConfig
dictConfig(settings.LOGGING)



def is_tarball(fn):
    logmsg('is_tarball: %s' % fn)
    types = filetype_short(fn)
    logmsg('filetypes:', types)
    for t in types:
        if t.startswith('POSIX tar archive'):
            return True
    return False

def get_tarball_files(fn):
    # create temp dir to extract tarfile.
    tempdir = tempfile.mkdtemp()
    cmd = 'tar xvf %s -C %s' % (fn, tempdir)
    #userlog('Extracting tarball...')
    (rtn, out, err) = run_command(cmd)
    if rtn:
        #userlog('Failed to un-tar file:\n' + err)
        #bailout(submission, 'failed to extract tar file')
        print 'failed to extract tar file'
        return None
    fns = out.strip('\n').split('\n')

    validpaths = []
    for fn in fns:
        path = os.path.join(tempdir, fn)
        logmsg('Path "%s"' % path)
        if not os.path.exists(path):
            logmsg('Path "%s" does not exist.' % path)
            continue
        if os.path.islink(path):
            logmsg('Path "%s" is a symlink.' % path)
            continue
        if os.path.isfile(path):
            validpaths.append(path)
        else:
            logmsg('Path "%s" is not a file.' % path)

    if len(validpaths) == 0:
        #userlog('Tar file contains no regular files.')
        #bailout(submission, "tar file contains no regular files.")
        #return -1
        logmsg('No real files in tar file')
        return None

    logmsg('Got %i paths.' % len(validpaths))
    return validpaths

def run_pnmfile(fn):
    cmd = 'pnmfile %s' % fn
    (filein, fileout) = os.popen2(cmd)
    filein.close()
    out = fileout.read().strip()
    logmsg('pnmfile output: ' + out)
    pat = re.compile(r'P(?P<pnmtype>[BGP])M .*, (?P<width>\d*) by (?P<height>\d*) *maxval (?P<maxval>\d*)')
    match = pat.search(out)
    if not match:
        logmsg('No match.')
        return None
    w = int(match.group('width'))
    h = int(match.group('height'))
    pnmtype = match.group('pnmtype')
    maxval = int(match.group('maxval'))
    logmsg('Type %s, w %i, h %i, maxval %i' % (pnmtype, w, h, maxval))
    return (w, h, pnmtype, maxval)


def dojob(userimage):
    job = Job(user_image=userimage)
    job.set_start_time()
    job.save()
    dirnm = job.make_dir()
    os.chdir(dirnm)
    print 'Creating and entering directory', dirnm
    axyfn = 'job.axy'
    sub = userimage.submission
    df = userimage.image.disk_file
    img = userimage.image

    # Build command-line arguments for the augment-xylist program, which
    # detects sources in the image and adds processing arguments to the header
    # to produce a "job.axy" file.
    slo,shi = sub.get_scale_bounds()
    # Note, this must match Job.get_wcs_file().
    wcsfile = 'wcs.fits'
    axyargs = {
        '--out': axyfn,
        '--image': df.get_path(),
        '--scale-low': slo,
        '--scale-high': shi,
        '--scale-units': sub.scale_units,
        '--wcs': wcsfile,
        '--rdls': 'rdls.fits',
        # Other things we might want include...
        #'--pixel-error':,
        # --use-sextractor
        # --ra, --dec, --radius
        # --invert
        # -g / --guess-scale: try to guess the image scale from the FITS headers
        # --crpix-center: set the WCS reference point to the image center
        # --crpix-x <pix>: set the WCS reference point to the given position
        # --crpix-y <pix>: set the WCS reference point to the given position
        # -w / --width <pixels>: specify the field width
        # -e / --height <pixels>: specify the field height
        # -X / --x-column <column-name>: the FITS column name
        # -Y / --y-column <column-name>
        }

    # UGLY
    if sub.parity == 0:
        axyargs['--parity'] = 'pos'
    elif sub.parity == 1:
        axyargs['--parity'] = 'neg'

    cmd = 'augment-xylist '
    cmd += ' '.join(k + ((v and (' ' + str(v))) or '') for (k,v) in axyargs.items())
    logmsg('running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return False

    logmsg('created axy file ' + axyfn)
    # shell into compute server...
    logfn = 'log'
    cmd = ('(echo %(jobid)s; '
           ' tar cf - --ignore-failed-read %(axyfile)s) | '
           'ssh -x -T %(sshconfig)s 2>>%(logfile)s | '
           'tar xf - --atime-preserve -m --exclude=%(axyfile)s '
           '>>%(logfile)s 2>&1' %
           dict(jobid='job-%i' % (job.id), axyfile=axyfn,
                sshconfig=settings.ssh_solver_config,
                logfile=logfn))
    print 'command:', cmd
    w = os.system(cmd)
    if not os.WIFEXITED(w):
        print 'Solver failed'
        return
    rtn = os.WEXITSTATUS(w)
    if rtn:
        logmsg('Solver failed with return value %i' % rtn)
        return

    logmsg('Solver completed successfully.')
    
    # Solved?
    wcsfn = os.path.join(dirnm, wcsfile)
    logmsg('Checking for WCS file %s' % wcsfn)
    if os.path.exists(wcsfn):
        logmsg('WCS file exists')
        # Parse the wcs.fits file
        wcs = Tan(wcsfn, 0)
        # Convert to database model...
        tan = TanWCS(crval1=wcs.crval[0], crval2=wcs.crval[1],
                     crpix1=wcs.crpix[0], crpix2=wcs.crpix[1], 
                     cd11=wcs.cd[0], cd12=wcs.cd[1],
                     cd21=wcs.cd[2], cd22=wcs.cd[3],
                     imagew=img.width, imageh=img.height)
        tan.save()
        logmsg('Created TanWCS: ' + str(tan))
        # Find bounds for the Calibration object.
        r0,r1,d0,d1 = wcs.radec_bounds()
        calib = Calibration(raw_tan=tan, ramin=r0, ramax=r1, decmin=d0, decmax=d1)
        calib.save()
        logmsg("Created Calibration " + str(calib))
        job.calibration = calib
        job.save()
        job.status = 'S'
        logmsg('Saved job %i' % job.id)
    else:
        job.status = 'F'
    job.set_end_time()
    job.save()


def dosub(sub):
    print 'Processing submission:', sub
    sub.set_processing_started()
    sub.save()
    origname = None
    if sub.disk_file is None:
        print 'Retrieving URL', sub.url
        (fn, headers) = urllib.urlretrieve(sub.url)
        print 'Wrote to file', fn
        df = DiskFile.from_file(fn)
        # Try to split the URL into a filename component and save it
        p = urlparse(sub.url)
        p = p.path
        if p:
            s = p.split('/')
            origname = s[-1]
            sub.orig_filename = origname
        df.save()
        sub.disk_file = df
        sub.save()
    else:
        origname = sub.original_filename

    # compressed .gz?
    df = sub.disk_file
    fn = df.get_path()
    f,tmpfn = tempfile.mkstemp()
    os.close(f)
    comp = image2pnm.uncompress_file(fn, tmpfn)
    if comp:
        print 'Input file compression: %s' % comp
        fn = tmpfn

    # This is sort of crazy -- look at python's 'gzip' and 'tarfile' modules.
    '''
    if is_tarball(fn):
        logmsg('file is tarball.')
        fns = get_tarball_files(fn)
        if fns is None:
            return

        for fn in fns:
            df = DiskFile.for_file(fn)
            df.save()
            logmsg('New diskfile ' + df)
        shutil.rmtree(tempdir)
        return True
    '''

    # create Image object

    # Is there already an Image for this DiskFile?
    try:
        img,created = Image.objects.get_or_create(disk_file=df)
    except Image.MultipleObjectsReturned:
        img = Image.objects.filter(disk_file=df)
        for i in range(1,len(img)):
            img[i].delete()
        img = img[0]
        created = False

    if created:
        #img.save()
        # defaults=dict(width=w, height=h))

        # FIXME -- move this code to Image?
        # Convert file to pnm to find its size.
        f,pnmfn = tempfile.mkstemp()
        os.close(f)
        logmsg('Converting %s to %s...\n' % (fn, pnmfn))
        (filetype, errstr) = image2pnm.image2pnm(fn, pnmfn)
        if errstr:
            logmsg('Error converting image file: %s' % errstr)
            return
        x = run_pnmfile(pnmfn)
        if x is None:
            print 'couldn\'t find image file size'
            return
        (w, h, pnmtype, maxval) = x
        logmsg('Type %s, w %i, h %i' % (pnmtype, w, h))
        img.width = w
        img.height = h
        img.save()

    # create UserImage object.
    uimg,created = UserImage.objects.get_or_create(submission=sub, image=img, user=sub.user,
                                                   defaults=dict(original_file_name=origname))
    if created:
        uimg.save()

    sub.set_processing_finished()
    sub.save()



def main():
    nthreads = 1

    pool = None
    if nthreads > 1:
        pool = multiprocessing.Pool(nthreads)

    # multiprocessing.Lock for django db?

    # Find Submissions that have been started but not finished;
    # reset the start times to null.
    oldsubs = Submission.objects.filter(processing_started__isnull=False,
                                        processing_finished__isnull=True)
    for sub in oldsubs:
        print 'Resetting the processing status of', sub
        sub.processing_started = None
        sub.save()

    oldjobs = Job.objects.filter(start_time__isnull=False,
                                 end_time__isnull=True)
    #for job in oldjobs:
    #    #print 'Resetting the processing status of', job
    #    #job.start_time = None
    #    #job.save()
    # FIXME -- really?
    oldjobs.delete()

    while True:
        print 'Checking for new Submissions'
        newsubs = Submission.objects.filter(processing_started__isnull=True)
        #newsubs = Submission.objects.all()
        print 'Found', newsubs.count(), 'unstarted submissions'

        print 'Checking for UserImages without Jobs'
        # Can't figure out how to do this... tried:
        #newuis = UserImage.objects.filter(jobs__count=0)
        #newuis = UserImage.objects.filter(jobs__exists=False)
        #newuis = UserImage.objects.filter(jobs__len=False)
        newuis = UserImage.objects.all()
        newuis = [ui for ui in newuis if not ui.jobs.exists()]
        # --> SELECT (1) AS "a" FROM "net_job" WHERE "net_job"."user_image_id" = 1  LIMIT 1; args=(1,)
        print 'Found', len(newuis), 'userimages without Jobs'

        if len(newsubs) + len(newuis) == 0:
            time.sleep(3)
            continue
        # FIXME -- order by user, etc
        for sub in newsubs:
            if pool:
                pool.apply_async(dosub, (sub,))
            else:
                dosub(sub)

        for ui in newuis:
            if pool:
                pool.apply_async(dojob, (ui,))
            else:
                dojob(ui)
            
    

if __name__ == '__main__':
    main()

