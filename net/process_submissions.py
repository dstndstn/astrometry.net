#! /usr/bin/env python

import os
import sys

# add .. to PYTHONPATH
path = os.path.realpath(__file__)
print 'Path', path
basedir = os.path.dirname(os.path.dirname(path))
print 'Adding basedir', basedir, 'to PYTHONPATH'
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
import math

import logging
#logging.basicConfig(format='%(message)s',
#                    level=logging.DEBUG)

from astrometry.util import image2pnm
from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command

from astrometry.util.util import Tan
from astrometry.util import util as anutil
from astrometry.util.pyfits_utils import *

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
from django.db import DatabaseError
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
    '''
    Create a MyLogger object that writes to a log file within a Job directory.
    '''
    logmsg("getlogger")
    logger = logging.getLogger('job.%i' % job.id)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(job.get_log_file2())
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return MyLogger(logger)

def try_dojob(job, userimage):
    jobdir = job.make_dir()
    log = create_job_logger(job)
    try:
        return dojob(job, userimage, log=log)
    except:
        print 'Caught exception while processing Job', job
        traceback.print_exc(None, sys.stdout)
        # FIXME -- job.set_status()...
        job.set_end_time()
        job.status = 'F'
        job.save()
        log.msg('Caught exception while processing Job', job.id)
        log.msg(traceback.format_exc(None))

def dojob(job, userimage, log=None):
    jobdir = job.get_dir()
    if log is None:
        log = create_job_logger(job)
    log.msg('Starting Job processing for', job)
    job.set_start_time()
    job.save()
    #os.chdir(dirnm) - not thread safe (working directory is global)!
    log.msg('Creating directory', jobdir)
    axyfn = 'job.axy'
    axypath = os.path.join(jobdir, axyfn)
    sub = userimage.submission
    log.msg('submission id', sub.id)
    df = userimage.image.disk_file
    img = userimage.image

    # Build command-line arguments for the augment-xylist program, which
    # detects sources in the image and adds processing arguments to the header
    # to produce a "job.axy" file.
    slo,shi = sub.get_scale_bounds()
    # Note, this must match Job.get_wcs_file().
    wcsfile = 'wcs.fits'
    axyflags = []
    axyargs = {
        '--out': axypath,
        '--scale-low': slo,
        '--scale-high': shi,
        '--scale-units': sub.scale_units,
        '--wcs': wcsfile,
        '--rdls': 'rdls.fits',
        '--pixel-error': sub.positional_error,
        '--ra': sub.center_ra,
        '--dec': sub.center_dec,
        '--radius': sub.radius,
        '--downsample': sub.downsample_factor,
        # tuning-up maybe fixed; if not, turn it off with:
        #'--odds-to-tune': 1e9,

        # Other things we might want include...
        # --use-sextractor
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

    if hasattr(img,'sourcelist'):
        # image is a source list; use --xylist
        axyargs['--xylist'] = img.sourcelist.get_fits_path()
        axyargs['--width'] = img.width
        axyargs['--height'] = img.height
    else:
        axyargs['--image'] = df.get_path()

    # UGLY
    if sub.parity == 0:
        axyargs['--parity'] = 'pos'
    elif sub.parity == 1:
        axyargs['--parity'] = 'neg'

    if sub.tweak_order == 0:
        axyflags.append('--no-tweak')
    else:
        axyargs['--tweak-order'] = '%i' % sub.tweak_order

    if sub.use_sextractor:
        axyflags.append('--use-sextractor')

    cmd = 'augment-xylist '
    for (k,v) in axyargs.items():
        if v:
            cmd += k + ' ' + str(v) + ' '
    for k in axyflags:
        cmd += k + ' '

    log.msg('running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        log.msg('out: ' + out)
        log.msg('err: ' + err)
        logmsg('augment-xylist failed: rtn val', rtn, 'err', err)
        raise Exception

    log.msg('created axy file', axypath)
    # shell into compute server...
    logfn = job.get_log_file()
    # the "tar" commands both use "-C" to chdir, and the ssh command
    # and redirect uses absolute paths.
    cmd = ('(echo %(jobid)s; '
           'tar cf - --ignore-failed-read -C %(jobdir)s %(axyfile)s) | '
           'ssh -x -T %(sshconfig)s 2>>%(logfile)s | '
           'tar xf - --atime-preserve -m --exclude=%(axyfile)s -C %(jobdir)s '
           '>>%(logfile)s 2>&1' %
           dict(jobid='job-%s-%i' % (settings.sitename, job.id),
                axyfile=axyfn, jobdir=jobdir,
                sshconfig=settings.ssh_solver_config,
                logfile=logfn))
    log.msg('command:', cmd)
    w = os.system(cmd)
    if not os.WIFEXITED(w):
        log.msg('Solver failed (sent signal?)')
        logmsg('Call to solver failed for job', job.id)
        raise Exception
    rtn = os.WEXITSTATUS(w)
    if rtn:
        log.msg('Solver failed with return value %i' % rtn)
        logmsg('Call to solver failed for job', job.id, 'with return val', rtn)
        raise Exception

    log.msg('Solver completed successfully.')

    # Solved?
    wcsfn = os.path.join(jobdir, wcsfile)
    log.msg('Checking for WCS file', wcsfn)
    if os.path.exists(wcsfn):
        log.msg('WCS file exists')
        # Parse the wcs.fits file
        wcs = Tan(wcsfn, 0)
        # Convert to database model...
        tan = TanWCS(crval1=wcs.crval[0], crval2=wcs.crval[1],
                     crpix1=wcs.crpix[0], crpix2=wcs.crpix[1],
                     cd11=wcs.cd[0], cd12=wcs.cd[1],
                     cd21=wcs.cd[2], cd22=wcs.cd[3],
                     imagew=img.width, imageh=img.height)
        tan.save()
        log.msg('Created TanWCS:', tan)

        # Find field's healpix nside and index
        ra, dec, radius = tan.get_center_radecradius()
        nside = anutil.healpix_nside_for_side_length_arcmin(radius*60)
        nside = int(2**round(math.log(nside, 2)))
        healpix = anutil.radecdegtohealpix(ra, dec, nside)
        sky_location, created = SkyLocation.objects.get_or_create(nside=nside, healpix=healpix)
        log.msg('SkyLocation:', sky_location)

        # Find bounds for the Calibration object.
        r0,r1,d0,d1 = wcs.radec_bounds()
        # Find cartesian coordinates
        ra *= math.pi/180
        dec *= math.pi/180
        tempr = math.cos(dec)
        x = tempr*math.cos(ra)
        y = tempr*math.sin(ra)
        z = math.sin(dec)
        r = radius/180*math.pi

        calib = Calibration(raw_tan=tan, ramin=r0, ramax=r1, decmin=d0, decmax=d1,
                            x=x,y=y,z=z,r=r,
                            sky_location=sky_location)
        calib.save()
        log.msg('Created Calibration', calib)
        job.calibration = calib
        job.save() # save calib before adding machine tags
        job.status = 'S'
        job.user_image.add_machine_tags(job)
        job.user_image.add_sky_objects(job)
    else:
        job.status = 'F'
    job.set_end_time()
    job.save()
    log.msg('Finished job', job.id)
    logmsg('Finished job',job.id)
    return job.id

def try_dosub(sub, max_retries):
    try:
        return dosub(sub)
    except DatabaseError as e:
        if (sub.processing_retries < max_retries):
            print 'Retrying processing Submission %s' % str(sub)
            sub.processing_retries += 1
            sub.save()
            return try_dosub(sub, max_retries)
        else:
            print 'Submission retry limit reached'
            sub.set_error_message(
                'Caught exception while processing Submission: ' +  str(sub) + '\n'
                + traceback.format_exc(None))
            sub.set_processing_finished()
            sub.save()
            return 'exception'
    except:
        print 'Caught exception while processing Submission', sub
        traceback.print_exc(None, sys.stdout)
        # FIXME -- sub.set_status()...
        sub.set_error_message(
            'Caught exception while processing Submission: ' +  str(sub) + '\n'
            + traceback.format_exc(None))
        sub.set_processing_finished()
        sub.save()
        logmsg('Caught exception while processing Submission ' + str(sub))
        logmsg('  ' + traceback.format_exc(None))
        return 'exception'

def dosub(sub):
    #sub.set_processing_really_started()
    #sub.save()
    #logmsg('sub license settings: commercial=%s, modifications=%s' % (
    #    sub.license.allow_commercial_use,
    #    sub.license.allow_modifications))
    if sub.disk_file is None:
        logmsg('Sub %i: retrieving URL' % (sub.id), sub.url)
        (fn, headers) = urllib.urlretrieve(sub.url)
        logmsg('Sub %i: wrote URL to file' % (sub.id), fn)
        df = DiskFile.from_file(fn, Image.ORIG_COLLECTION)
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

    else:
        logmsg('uploaded disk file for this submission is ' + str(sub.disk_file))

    df = sub.disk_file
    fn = df.get_path()
    logmsg('DiskFile path ' + fn)

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
        df = DiskFile.from_file(tempfn, 'uploaded-gunzip')
        i = original_filename.find('.gz')
        if i != -1:
            original_filename = original_filename[:i]
        logmsg('extracted gzip file %s' % original_filename)
        fn = tempfn
    except:
        # not a gzip file
        pass

    if tarfile.is_tarfile(fn):
        logmsg('File %s: tarball' % fn)
        tar = tarfile.open(fn)
        dirnm = tempfile.mkdtemp()
        for tarinfo in tar.getmembers():
            if tarinfo.isfile():
                logmsg('extracting file %s' % tarinfo.name)
                tar.extract(tarinfo, dirnm)
                tempfn = os.path.join(dirnm, tarinfo.name)
                df = DiskFile.from_file(tempfn, 'uploaded-untar')
                # create Image object
                img = get_or_create_image(df)
                # create UserImage object.
                if img:
                    create_user_image(sub, img, tarinfo.name)

        tar.close()
        shutil.rmtree(dirnm, ignore_errors=True)
    else:
        # assume file is single image
        logmsg('File %s: single file' % fn)
        # create Image object
        img = get_or_create_image(df)
        logmsg('File %s: created Image %s' % (fn, str(img)))
        # create UserImage object.
        if img:
            logmsg('File %s: Image id %i' % (fn, img.id))
            uimg = create_user_image(sub, img, original_filename)
            logmsg('Image %i: created UserImage %i' % (img.id, uimg.id))

    sub.set_processing_finished()
    sub.save()
    return sub.id

def create_user_image(sub, img, original_filename):
    license, created = License.objects.get_or_create(
        default_license=sub.user.get_profile().default_license,
        allow_modifications = sub.license.allow_modifications,
        allow_commercial_use = sub.license.allow_commercial_use,
    )
    comment_receiver = CommentReceiver.objects.create()
    uimg,created = UserImage.objects.get_or_create(
        submission=sub,
        image=img,
        user=sub.user,
        license=license,
        comment_receiver=comment_receiver,
        defaults=dict(original_file_name=original_filename,
                     publicly_visible = sub.publicly_visible))
    if sub.album:
        sub.album.user_images.add(uimg)
    return uimg

def get_or_create_image(df):
    # Is there already an Image for this DiskFile?
    try:
        img = Image.objects.get(disk_file=df, display_image__isnull=False, thumbnail__isnull=False)
    except Image.MultipleObjectsReturned:
        logmsg("multiple found")
        imgs = Image.objects.filter(disk_file=df, display_image__isnull=False, thumbnail__isnull=False)
        for i in range(1,len(imgs)):
            imgs[i].delete()
        img = imgs[0]
    except Image.DoesNotExist:
        # try to create image assume disk file is an image file (png, jpg, etc)
        img = create_image(df)
        logmsg('img = ' + str(img))
        if img is None:
            # try to create sourcelist image
            img = create_source_list(df)

        if img:
            # cache
            img.get_thumbnail()
            img.get_display_image()
            img.save()
        else:
            raise Exception('This file\'s type is not supported.')
    return img


def create_image(df):
    img = None
    try:
        img = Image(disk_file=df)
        # FIXME -- move this code to Image?
        # Convert file to pnm to find its size.
        pnmfn = img.get_pnm_path()
        x = run_pnmfile(pnmfn)
        if x is None:
            raise RuntimeError('Could not find image file size')
        (w, h, pnmtype, maxval) = x
        logmsg('Type %s, w %i, h %i' % (pnmtype, w, h))
        img.width = w
        img.height = h
        img.save()
    except:
        logmsg('file is not an image file')
        img = None
    return img

def create_source_list(df):
    img = None
    fits = None
    source_type = None
    try:
        # see if disk file is a fits list
        fits = fits_table(str(df.get_path()))
        source_type = 'fits'
    except:
        logmsg('file is not a fits table')
        # otherwise, check to see if it is a text list
        try:
            fitsfn = get_temp_file()

            text_file = open(str(df.get_path()))
            text = text_file.read()
            text_file.close()

            # add x y header
            # potential hack, assumes it doesn't exist...
            text = "# x y\n" + text

            text_table = text_table_fields("", text=text)
            text_table.write_to(fitsfn)
            logmsg("Creating fits table from text list")

            fits = fits_table(fitsfn)
            source_type = 'text'
        except Exception as e:
            import traceback
            logmsg('Traceback:\n' + traceback.format_exc())
            logmsg('fitsfn: %s' % fitsfn)
            logmsg(e)
            logmsg('file is not a text list')

    if fits:
        try:
            img = SourceList(disk_file=df, source_type=source_type)
            # w = fits.x.max()-fits.x.min()
            # h = fits.y.max()-fits.y.min()
            # w = int(w)
            # h = int(h)
            w = int(math.ceil(fits.x.max()))
            h = int(math.ceil(fits.y.max()))
            logmsg('w %i, h %i' % (w, h))
            img.width = w
            img.height = h
            img.save()
        except Exception as e:
            logmsg(e)
            img = None
            raise e

    return img

## DEBUG
def sub_callback(result):
    print 'Submission callback: Result:', result
def job_callback(result):
    print 'Job callback: Result:', result


def main(dojob_nthreads, dosub_nthreads, refresh_rate, max_sub_retries):
    dojob_pool = None
    dosub_pool = None
    if dojob_nthreads > 1:
        print 'Processing jobs with %d threads' % dojob_nthreads
        dojob_pool = multiprocessing.Pool(processes=dojob_nthreads)
    if dosub_nthreads > 1:
        print 'Processing submissions with %d threads' % dosub_nthreads
        dojob_pool = multiprocessing.Pool(processes=dojob_nthreads)
        dosub_pool = multiprocessing.Pool(processes=dosub_nthreads)

    print 'Refresh rate: %.1f seconds' % refresh_rate
    print 'Submission processing retry limit: %d' % max_sub_retries

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
    jobresults = []

    #
    me = ProcessSubmissions(pid=os.getpid())
    me.set_watchdog()
    me.save()

    lastsubs = []
    lastjobs = []

    while True:
        me.set_watchdog()
        me.save()

        print

        #print 'Checking for new Submissions'
        newsubs = Submission.objects.filter(processing_started__isnull=True)
        if newsubs.count():
            print 'Found', newsubs.count(), 'unstarted Submissions:', [s.id for s in newsubs]

        #print 'Checking for UserImages without Jobs'
        all_user_images = UserImage.objects.annotate(job_count=Count('jobs'))
        newuis = all_user_images.filter(job_count=0)
        if newuis.count():
            print 'Found', len(newuis), 'UserImages without Jobs:', [u.id for u in newuis]

        runsubs = me.subs.filter(finished=False)
        if subresults != lastsubs:
            print 'Submissions running:', len(subresults)
            lastsubs = subresults
        for sid,res in subresults:
            print '  Submission id', sid, 'ready:', res.ready(),
            if res.ready():
                subresults.remove((sid,res))
                print 'success:', res.successful(),

                qs = runsubs.get(submission__id=sid)
                qs.finished = True
                qs.success = res.successful()
                qs.save()

                if res.successful():
                    print 'result:', res.get(),
            print

        runjobs = me.jobs.filter(finished=False)
        if jobresults != lastjobs:
            print 'Jobs running:', len(jobresults)
            lastjobs = jobresults
        for jid,res in jobresults:
            print '  Job id', jid, 'ready:', res.ready(),
            if res.ready():
                jobresults.remove((jid,res))
                print 'success:', res.successful(),

                qj = runjobs.get(job__id=jid)
                qj.finished = True
                qj.success = res.successful()
                qj.save()

                if res.successful():
                    print 'result:', res.get(),
            print

        if (len(newsubs) + len(newuis)) == 0:
            time.sleep(refresh_rate)
            continue

        # FIXME -- order by user, etc

        for sub in newsubs:
            print 'Enqueuing submission:', sub
            sub.set_processing_started()
            sub.save()

            qs = QueuedSubmission(procsub=me, submission=sub)
            qs.save()

            if dosub_pool:
                res = dosub_pool.apply_async(
                    try_dosub,
                    (sub, max_sub_retries),
                    callback=sub_callback
                )
                subresults.append((sub.id, res))
            else:
                try_dosub(sub, max_sub_retries)


        for userimage in newuis:
            job = Job(user_image=userimage)
            job.set_queued_time()
            job.save()

            qj = QueuedJob(procsub=me, job=job)
            qj.save()

            if dojob_pool:
                res = dojob_pool.apply_async(try_dojob, (job, userimage),
                                             callback=job_callback)
                jobresults.append((job.id, res))
            else:
                dojob(job, userimage)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--jobthreads', '-j', dest='jobthreads', type='int',
                      default=3, help='Set the number of threads to process jobs')
    parser.add_option('--subthreads', '-s', dest='subthreads', type='int',
                      default=2, help='Set the number of threads to process submissions')
    parser.add_option('--maxsubretries', '-m', dest='maxsubretries', type='int',
                      default=20, help='Set the maximum number of times to retry processing a submission')
    parser.add_option('--refreshrate', '-r', dest='refreshrate', type='float',
                      default=5, help='Set how often to check for new jobs and submissions (in seconds)')
    opt,args = parser.parse_args()

    main(opt.jobthreads, opt.subthreads, opt.refreshrate, opt.maxsubretries)
