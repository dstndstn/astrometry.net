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
import tarfile
import gzip
import zipfile

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
    'level': 'INFO',
    'propagate': True,
}
from astrometry.net.models import *
from log import *

from django.utils.log import dictConfig
from django.db.models import Count
from django.db.models import Q

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

class MyLogger(object):
    def __init__(self, logger):
        self.logger = logger
    def debug(self, *args):
        return self.logger.debug(' '.join(str(x) for x in args))
    def info(self, *args):
        return self.logger.info(' '.join(str(x) for x in args))
    msg = info

def create_job_logger(job):
    logmsg("getlogger")
    logger = logging.getLogger('job.%i' % job.id)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(job.get_log_file2())
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return MyLogger(logger)


def makejobs(userimages, job_queue):
    for userimage in userimages:
        job = Job(user_image=userimage)
        job.set_queued_time()
        job.save()
        job_queue.put({
            'job':job,
            'userimage':userimage
        })


def dojob(job,userimage):
    logmsg("dojob")
    dirnm = job.make_dir()
    log = create_job_logger(job)
    log.msg('Starting Job processing for', job)
    logmsg("createlogger")
    job.set_start_time()
    job.save()
    #os.chdir(dirnm) - not thread safe (working directory is global)!
    log.msg('Creating directory', dirnm)
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
        '--out': os.path.join(dirnm, axyfn),
        '--image': df.get_path(),
        '--scale-low': slo,
        '--scale-high': shi,
        '--scale-units': sub.scale_units,
        '--wcs': wcsfile,
        '--rdls': 'rdls.fits',
        '--pixel-error': sub.positional_error,
        '--ra': sub.center_ra,
        '--dec': sub.center_dec,
        '--radius': sub.radius,
        '--downsample': sub.downsample_factor

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
    for (k,v) in axyargs.items():
        if v:
            cmd += k + ' ' + str(v) + ' '

    logmsg('running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return False

    logmsg('created axy file ' + axyfn)
    # shell into compute server...
    logfn = os.path.join(dirnm, 'log')
    cmd = ('(echo %(jobid)s; '
           ' tar cf - --ignore-failed-read -C %(dirnm)s %(axyfile)s) | '
           'ssh -x -T %(sshconfig)s 2>>%(logfile)s | '
           'tar xf - --atime-preserve -m --exclude=%(axyfile)s -C %(dirnm)s '
           '>>%(logfile)s 2>&1' %
           dict(jobid='job-%s-%i' % (settings.sitename, job.id),
                axyfile=axyfn, dirnm=dirnm,
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
        job.user_image.add_machine_tags()
        logmsg('Saved job %i' % job.id)
    else:
        job.status = 'F'
    job.set_end_time()
    job.save()


def queue_subs(newsubs, sub_queue):
    for sub in newsubs:
        print 'Enqueuing submission:', sub
        sub.set_processing_started()
        sub.save()
        sub_queue.put(sub)

def dosub(sub):
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
            sub.original_filename = origname
        df.save()
        sub.disk_file = df
        sub.save()

    # compressed .gz
    df = sub.disk_file
    fn = df.get_path()

    '''
    f,tmpfn = tempfile.mkstemp()
    os.close(f)
    comp = image2pnm.uncompress_file(fn, tmpfn)
    if comp:
        print 'Input file compression: %s' % comp
        fn = tmpfn
    '''
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

    if tarfile.is_tarfile(fn):
        logmsg('file is a tarball')
        tar = tarfile.open(fn)
        dirnm = tempfile.mkdtemp()
        for tarinfo in tar.getmembers():
            if tarinfo.isfile():
                logmsg('extracting file %s' % tarinfo.name)
                tar.extract(tarinfo, dirnm)
                tempfn = os.path.join(dirnm, tarinfo.name)
                df = DiskFile.from_file(tempfn)
                # create Image object
                img = get_or_create_image(df)
                # create UserImage object.
                uimg,created = UserImage.objects.get_or_create(submission=sub, image=img, user=sub.user,
                                                               defaults=dict(original_file_name=tarinfo.name))
        tar.close()
        os.remove(dirnm)
    else:
        original_filename = sub.original_filename
        # check if file is a gzipped file
        try:
            gzip_file = gzip.open(fn)
            f,tempfn = tempfile.mkstemp()
            os.close(f)
            f = open(tempfn,'wb')
            # should fail on the following line if not a gzip file
            f.write(gzip_file.read())
            f.close()
            gzip_file.close()
            df = DiskFile.from_file(tempfn)
            i = original_filename.find('.gz')
            if i != -1:
                original_filename = original_filename[:i]
            logmsg('extracted gzip file %s' % original_filename)
        except:
            # not a gzip file
            pass
         
        # assume file is single image
        logmsg('single file')
        # create Image object
        img = get_or_create_image(df)
        # create UserImage object.
        uimg,created = UserImage.objects.get_or_create(submission=sub, image=img, user=sub.user,
                                                       defaults=dict(original_file_name=original_filename))

    sub.set_processing_finished()
    sub.save()
    return sub.id

def get_or_create_image(df):
    # Is there already an Image for this DiskFile?
    try:
        img,created = Image.objects.get_or_create(disk_file=df)
    except Image.MultipleObjectsReturned:
        img = Image.objects.filter(disk_file=df)
        for i in range(1,len(img)):
            img[i].delete()
        img = img[0]
        created = False

    fn = df.get_path()
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
        # cache
        img.get_thumbnail()
        img.get_display_image()
        img.save()

    return img

def main():
    dosub_queue = multiprocessing.Queue()
    job_queue = multiprocessing.Queue()
    dojob_nthreads = 3
    dosub_nthreads = 2

    dojob_pool = None
    dosub_pool = None
    if dojob_nthreads > 1:
        dojob_pool = multiprocessing.Pool(processes=dojob_nthreads)
    if dosub_nthreads > 1:
        dosub_pool = multiprocessing.Pool(processes=dosub_nthreads)

    # multiprocessing.Lock for django db?

    # Find Submissions that have been started but not finished;
    # reset the start times to null.
    oldsubs = Submission.objects.filter(processing_started__isnull=False,
                                        processing_finished__isnull=True)
    for sub in oldsubs:
        print 'Resetting the processing status of', sub
        sub.processing_started = None
        sub.save()

    oldjobs = Job.objects.filter(Q(end_time__isnull=True),
                                 Q(start_time__isnull=False) |
                                 Q(queued_time__isnull=False))
    #for job in oldjobs:
    #    #print 'Resetting the processing status of', job
    #    #job.start_time = None
    #    #job.save()
    # FIXME -- really?
    oldjobs.delete()

    subresults = []

    while True:
        print 'Checking for new Submissions'
        newsubs = Submission.objects.filter(processing_started__isnull=True)
        print 'Found', newsubs.count(), 'unstarted submissions'

        print 'Checking for UserImages without Jobs'
        all_user_images = UserImage.objects.annotate(job_count=Count('jobs'))
        newuis = all_user_images.filter(job_count=0)
        print 'Found', len(newuis), 'userimages without Jobs'

        print 'Submission async results:', subresults

        print 'dosub_queue:', dosub_queue
        print 'job_queue:', job_queue

        if ((len(newsubs) + len(newuis) == 0) and
            dosub_queue.empty() and
            job_queue.empty()):
            time.sleep(5)
            continue

        # FIXME -- order by user, etc
        queue_subs(newsubs,dosub_queue)

        ''' dstn asks:
        What is the point of using queues here?  queue_subs just adds everything
        in 'newsubs' to the queue, and then you immediately pull them out and
        submit them to the pool for processing, all in this main thread.

        I agree that stamping the Submission/Job before giving it to the pool
        is necessary, but that is a separate issue.
        '''

        def sub_callback(result):
            print 'Submission callback: Result:', result

        while not dosub_queue.empty():
            try:
                sub = dosub_queue.get()
                if dosub_pool:
                    res = dosub_pool.apply_async(dosub, (sub,), callback=sub_callback)
                    subresults.append(res)
                else:
                    dosub(sub)
            except multiprocessing.Queue.Empty as e:
                pass

        makejobs(newuis, job_queue)

        while not job_queue.empty():
            try:
                job_ui = job_queue.get()
                if dojob_pool:
                    dojob_pool.apply_async(dojob, (
                        job_ui['job'],
                        job_ui['userimage'],
                        )
                    )
                else:
                    dojob(job_ui['job'], job_ui['userimage'])
            except multiprocessing.Queue.Empty as e:
                pass

if __name__ == '__main__':
    main()

