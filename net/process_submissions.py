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
        log.msg('Caught exception while processing Job', job.id)
        log.msg(traceback.format_exc(None))

def dojob(job, userimage, log=None):
    jobdir = job.make_dir()
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
        '--downsample': sub.downsample_factor

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

    cmd = 'augment-xylist '
    for (k,v) in axyargs.items():
        if v:
            cmd += k + ' ' + str(v) + ' '

    log.msg('running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        log.msg('out: ' + out)
        log.msg('err: ' + err)
        logmsg('augment-xylist failed: rtn val', rtn, 'err', err)
        return False

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
        return
    rtn = os.WEXITSTATUS(w)
    if rtn:
        log.msg('Solver failed with return value %i' % rtn)
        logmsg('Call to solver failed for job', job.id, 'with return val', rtn)
        return

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
        # Find bounds for the Calibration object.
        r0,r1,d0,d1 = wcs.radec_bounds()
        calib = Calibration(raw_tan=tan, ramin=r0, ramax=r1, decmin=d0, decmax=d1)
        calib.save()
        log.msg('Created Calibration', calib)
        job.calibration = calib
        job.save() # save calib before adding machine tags
        job.status = 'S'
        job.user_image.add_machine_tags(job)
    else:
        job.status = 'F'
    job.set_end_time()
    job.save()
    log.msg('Finished job', job.id)
    logmsg('Finished job',job.id)
    return job.id

def try_dosub(sub):
    try:
        return dosub(sub)
    except:
        print 'Caught exception while processing Submission', sub
        traceback.print_exc(None, sys.stdout)
        # FIXME -- sub.set_status()...
        sub.set_processing_finished()
        sub.save()
        return 'exception'

def dosub(sub):
    logmsg('sub license settings: commercial=%s, modifications=%s' % (
        sub.allow_commercial_use,
        sub.allow_modifications))
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
                if sub.source_type == 'image':
                    img = get_or_create_image(df)
                else:
                    img = get_or_create_source_list(df, sub.source_type)
                # create UserImage object.
                if img:
                    uimg,created = UserImage.objects.get_or_create(
                        submission=sub,
                        image=img,
                        user=sub.user,
                        defaults=dict(original_file_name=tarinfo.name,
                                      allow_modifications = sub.allow_modifications,
                                      allow_commercial_use = sub.allow_commercial_use,
                                      publicly_visible = sub.publicly_visible))
                    if sub.album:
                        sub.album.user_images.add(uimg)

                #os.remove(tempfn)
        tar.close()
        os.removedirs(dirnm)
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
        if sub.source_type == 'image':
            img = get_or_create_image(df)
        else:
            img = get_or_create_source_list(df, sub.source_type)
        # create UserImage object.
        if img:
            uimg,created = UserImage.objects.get_or_create(
                submission=sub, image=img, user=sub.user,
                defaults=dict(original_file_name=original_filename,
                             allow_modifications = sub.allow_modifications,
                             allow_commercial_use = sub.allow_commercial_use,
                             publicly_visible = sub.publicly_visible))
            if sub.album:
                sub.album.user_images.add(uimg)
            #uimg.save()

    sub.set_processing_finished()
    sub.save()
    return sub.id

def get_or_create_image(df):
    # Is there already an Image for this DiskFile?
    try:
        img,created = Image.objects.get_or_create(disk_file=df, sourcelist__isnull=True)
    except Image.MultipleObjectsReturned:
        logmsg("multiple found")
        img = Image.objects.filter(disk_file=df, sourelist_isnull=True)
        for i in range(1,len(img)):
            img[i].delete()
        img = img[0]
        created = False

    if created:
        try:
            # FIXME -- move this code to Image?
            # Convert file to pnm to find its size.
            fn = df.get_path()
            f,pnmfn = tempfile.mkstemp()
            os.close(f)
            logmsg('Converting %s to %s...\n' % (fn, pnmfn))
            (filetype, errstr) = image2pnm.image2pnm(fn, pnmfn)
            if errstr:
                raise RuntimeError('Error converting image file: %s' % errstr)
            x = run_pnmfile(pnmfn)
            if x is None:
                raise RuntimeError('Could not find image file size')
            (w, h, pnmtype, maxval) = x
            logmsg('Type %s, w %i, h %i' % (pnmtype, w, h))
            img.width = w
            img.height = h
            img.save()
            # cache
            img.get_thumbnail()
            img.get_display_image()
            img.save()
        except Exception as e:
            # delete image if anything fails
            logmsg(e)
            logmsg('deleting Image')
            img.delete()
            img = None
        except:
            # FIXME (something throws a SystemExit..)
            # delete image if anything fails
            logmsg(sys.exc_info()[0])
            logmsg('deleting Image')
            img.delete()
            img = None
    return img

def get_or_create_source_list(df, source_type):
    # Is there already an SourceList for this DiskFile?
    try:
        img,created = SourceList.objects.get_or_create(disk_file=df, 
                                                       source_type=source_type,
                                                       display_image__isnull=False)
    except SourceList.MultipleObjectsReturned:
        img = SourceList.objects.filter(disk_file=df,
                                        source_type=source_type,
                                        display_image__isnull=False)
        for i in range(1,len(img)):
            img[i].delete()
        img = img[0]
        created = False

    if created:
        try:
            fits = img.get_fits_table()
            w = fits.x.max()-fits.x.min()
            h = fits.y.max()-fits.y.min()
            w = int(1.01*w)
            h = int(1.01*h)
            logmsg('w %i, h %i' % (w, h))
            img.width = w
            img.height = h
            img.save()
            # cache
            img.get_thumbnail()
            img.get_display_image()
            img.save()
        except Exception as e:
            # delete image if anything fails
            logmsg(e)
            logmsg('deleting SourceList')
            img.delete()
            img = None
    
    return img
    
## DEBUG
def sub_callback(result):
    print 'Submission callback: Result:', result
def job_callback(result):
    print 'Job callback: Result:', result


def main(dojob_nthreads, dosub_nthreads, refresh_rate):
    dojob_pool = None
    dosub_pool = None
    if dojob_nthreads > 1:
        print 'Processing jobs with %d threads' % dojob_nthreads
        dojob_pool = multiprocessing.Pool(processes=dojob_nthreads)
    if dosub_nthreads > 1:
        print 'Processing submissions with %d threads' % dosub_nthreads
        dojob_pool = multiprocessing.Pool(processes=dojob_nthreads)
        dosub_pool = multiprocessing.Pool(processes=dosub_nthreads)

    print 'Refresh rate: %f seconds' % refresh_rate

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

    while True:
        me.set_watchdog()
        me.save()

        print 'Checking for new Submissions'
        newsubs = Submission.objects.filter(processing_started__isnull=True)
        print 'Found', newsubs.count(), 'unstarted submissions'

        print 'Checking for UserImages without Jobs'
        all_user_images = UserImage.objects.annotate(job_count=Count('jobs'))
        newuis = all_user_images.filter(job_count=0)
        print 'Found', len(newuis), 'userimages without Jobs'

        runsubs = me.subs.filter(finished=False)
        print 'Submissions running:', len(subresults)
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
        print 'Jobs running:', len(jobresults)
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
                res = dosub_pool.apply_async(try_dosub, (sub,),
                                             callback=sub_callback)
                subresults.append((sub.id, res))
            else:
                dosub(sub)


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
    parser.add_option('--refreshrate', '-r', dest='refreshrate', type='float',
                      default=5, help='Set how often to check for new jobs and submissions (in seconds)')
    opt,args = parser.parse_args()

    main(opt.jobthreads, opt.subthreads, opt.refreshrate)

