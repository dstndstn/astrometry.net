import random
import os
import errno
import hashlib
import shutil
import tempfile
from datetime import datetime

import numpy as np

from django.db import models
from django.db.models import Q
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.core.urlresolvers import reverse

from astrometry.net.settings import *
from wcs import *
from log import *

from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring
from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command
from astrometry.util.pyfits_utils import *
from astrometry.util.image2pnm import image2pnm
from astrometry.util import util as anutil
from astrometry.net.tmpfile import *

import PIL.Image, PIL.ImageDraw

from urllib2 import urlopen

from astrometry.net.abstract_models import *

### Admin view -- running Submissions and Jobs

# = dt.total_seconds() in python 2.7
class DuplicateSubmissionException(Exception):
    pass

class LicenseManager(models.Manager):
    def get_or_create(self, default_license=None, *args, **kwargs):
        return_value = None
        default_replaced_license = License(
            allow_commercial_use=kwargs.get('allow_commercial_use', 'd'),
            allow_modifications=kwargs.get('allow_modifications', 'd')
        )
        default_replaced_license.replace_license_default(default_license)
        kwargs = {
            'allow_commercial_use': default_replaced_license.allow_commercial_use,
            'allow_modifications': default_replaced_license.allow_modifications,
        }
        try:
            return_value = super(LicenseManager, self).get_or_create(*args, **kwargs)
        except MultipleObjectsReturned:
            license = License.objects.filter(**kwargs)[0]
            return_value = (license, False)

        return return_value
            

class License(models.Model):
    YES_NO = (
        ('y','yes'),
        ('n','no'),
        ('d','use default'),
    )
    YES_SA_NO = (
        ('y','yes'),
        ('sa','yes, but share alike'),
        ('n','no'),
        ('d','use default'),
    )

    objects = LicenseManager()

    # CC "answer" data
    allow_commercial_use = models.CharField(choices=YES_NO, max_length=1,
        default='d')
    allow_modifications = models.CharField(choices=YES_SA_NO, max_length=2,
        default='d')

    # CC issued license
    license_name = models.CharField(max_length=1024)
    license_uri = models.CharField(max_length=1024)

    # attribution and other optional fields here


    def get_license_uri(self):
        if self.license_uri == '':
            self.save()
        return self.license_uri

    def get_license_name(self):
        if self.license_name == '':
            self.save()
        return self.license_name

    # replaces 'd' with actual setting from default license
    def replace_license_default(self, default):
        if self.allow_commercial_use == 'd':
            self.allow_commercial_use = default.allow_commercial_use
        if self.allow_modifications == 'd':
            self.allow_modifications = default.allow_modifications

    def get_license_xml(self):
        try:
            allow_commercial_use = self.allow_commercial_use
            allow_modifications = self.allow_modifications

            url = (
                'http://api.creativecommons.org/rest/1.5/license/standard/get?commercial=%s&derivatives=%s&jurisdiction=' %
                (allow_commercial_use,
                allow_modifications,)
            )
            logmsg("getting license via url: %s" % url)
            f = urllib2.urlopen(url)
            xml = f.read()
            f.close()
            return xml
        except Exception as e:
            logmsg('error getting license xml: %s' % str(e))
            raise e

    # uses CC "answer" data to retrieve CC issued license data
    def get_license_name_uri(self):
        def get_text(node_list):
            rc = []
            for node in node_list:
                if node.nodeType == node.TEXT_NODE:
                    rc.append(node.data)
            return ''.join(rc)
        try:
            license_xml = self.get_license_xml()
            license_doc = xml.dom.minidom.parseString(license_xml)
            self.license_name = get_text(license_doc.getElementsByTagName('license-name')[0].childNodes)
            self.license_uri = get_text(license_doc.getElementsByTagName('license-uri')[0].childNodes)
            # can add rdf stuff here if we want..
            
        except Exception as e:
            logmsg('error getting issued license data: %s' % str(e))

    # provide cascading default replacement; replace instances of "use default"
    # with the default value specified by the default keyword argument, and
    # replace instances of "use default" from the keyword arg with the site
    # wide license defaults
    def save(self, default_license=None, *args, **kwargs):
        if default_license != None:
            self.replace_license_default(default_license)

        self.replace_license_default(License.get_default())
        self.get_license_name_uri()
        return super(License, self).save(*args,**kwargs)

    @staticmethod
    def get_default():
        # DEFAULT_LICENSE_ID  defined in settings_common.py
        return License.objects.get(pk=DEFAULT_LICENSE_ID)



class CommentReceiver(models.Model):
    owner = models.ForeignKey(User, null=True)

    # Reverse mappings:
    #   comments -> Comment



class Flag(models.Model):
    class Meta:
        ordering = ['name']

    name = models.CharField(max_length=56, primary_key=True)
    explanation = models.CharField(max_length=2048, blank=True)

    def input_name(self):
        return 'flag-%s' % self.name


def dtsec(dt):
    return (dt.microseconds + (dt.seconds + dt.days * 24. * 3600.) * 10.**6) / 10.**6



class ProcessSubmissions(models.Model):
    pid = models.IntegerField()
    watchdog = models.DateTimeField(null=True)
    # subs
    # jobs

    def set_watchdog(self):
        self.watchdog = datetime.now()

    def count_queued_subs(self):
        return self.subs.filter(finished=False,
                                submission__processing_started__isnull=True).count()

    def count_running_subs(self):
        return self.subs.filter(finished=False,
                                submission__processing_started__isnull=False).count()

    def count_queued_jobs(self):
        return self.jobs.filter(finished=False,
                                job__start_time__isnull=True).count()

    def count_running_jobs(self):
        return self.jobs.filter(finished=False,
                                job__start_time__isnull=False).count()

    def watchdog_ago(self):
        return datetime.now() - self.watchdog
    def watchdog_sec_ago(self):
        dt = self.watchdog_ago()
        sec = dtsec(dt)
        return '%i' % int(round(sec))


class QueuedThing(models.Model):
    finished = models.BooleanField()
    success = models.BooleanField()

    class Meta:
        abstract = True

    def get_time_string(self, t):
        if t is None:
            return '-'
        t = t.replace(microsecond=0)
        return t.isoformat() + ' (%i sec ago)' % dtsec(datetime.now() - t)

class QueuedSubmission(QueuedThing):
    procsub = models.ForeignKey('ProcessSubmissions', related_name='subs')
    submission = models.ForeignKey('Submission')
    def get_start_time_string(self):
        return self.get_time_string(self.submission.processing_started)
    def get_end_time_string(self):
        return self.get_time_string(self.submission.processing_finished)
        

class QueuedJob(QueuedThing):
    procsub = models.ForeignKey('ProcessSubmissions', related_name='jobs')
    job = models.ForeignKey('Job')
    def get_start_time_string(self):
        return self.get_time_string(self.job.start_time)
    def get_end_time_string(self):
        return self.get_time_string(self.job.end_time)

###


class DiskFile(models.Model):
    DEFAULT_COLLECTION = 'misc'

    collection = models.CharField(max_length=40,
                                  default=DEFAULT_COLLECTION)
    file_hash = models.CharField(max_length=40, unique=True, primary_key=True)
    size = models.PositiveIntegerField()
    file_type = models.CharField(max_length=256, null=True)

    # Reverse mappings:
    #  image_set -> Image
    #  submissions -> Submission
    #  cachedfile_set -> CachedFile ?
    

    def __str__(self):
        return 'DiskFile: %s, size %i, type %s, coll %s' % (self.file_hash, self.size, self.file_type, self.collection)

    def is_fits_image(self):
        return self.file_type == 'FITS image data'

    def set_size_and_file_type(self):
        fn = self.get_path()
        st = os.stat(fn)
        self.size = st.st_size
        filetypes = filetype_short(fn)
        self.file_type = ';'.join(filetypes)

    def get_file_types(self):
        return self.file_type.split(';')

    def OLD_get_path(self):
        h = self.file_hash
        return os.path.join('old-data', h[0:2], h[2:4], h[4:6], h)

    def get_path(self):
        return DiskFile.get_file_path(self.file_hash, self.collection)

    def make_dirs(self):
        return DiskFile.make_dirs_for(self.file_hash, self.collection)

    @staticmethod
    def get_file_directory(file_hash_digest,
                               collection=DEFAULT_COLLECTION):
        return os.path.join(DATADIR,
                            collection,
                            file_hash_digest[:3])

    @staticmethod
    def get_file_path(file_hash_digest,
                          collection=DEFAULT_COLLECTION):
        file_path = DiskFile.get_file_directory(file_hash_digest, collection)
        file_path = os.path.join(file_path, file_hash_digest)
        return file_path

    @staticmethod
    def make_dirs_for(file_hash_digest,
                      collection=DEFAULT_COLLECTION):
        file_directory = DiskFile.get_file_directory(file_hash_digest,
                                                     collection)
        try:
            os.makedirs(file_directory)
        except OSError as e:
            # we don't care if the directory already exists
            if e.errno == errno.EEXIST:
                pass
            else: raise

    @staticmethod
    def from_file(filename,
                  collection=DEFAULT_COLLECTION,
                  hashkey=None):
        if hashkey is None:
            file_hash = DiskFile.get_hash()
            f = open(filename)
            while True:
                s = f.read(8096)
                if not len(s):
                    # EOF
                    break
                file_hash.update(s)
            hashkey = file_hash.hexdigest()

        df,created = DiskFile.objects.get_or_create(
            file_hash=hashkey,
            defaults=dict(size=0, file_type='', collection=collection))
        if created:
            try:
                # move it into place
                df.make_dirs()
                shutil.move(filename, df.get_path())
                df.set_size_and_file_type()
                df.save()
            except:
                df.delete()
                raise
        return df

    @staticmethod
    def get_hash():
        return hashlib.sha1()

class CachedFile(models.Model):
    DEFAULT_COLLECTION = 'cached'

    disk_file = models.ForeignKey(DiskFile)
    key = models.CharField(max_length=64, unique=True, primary_key=True)

    @staticmethod
    def get(key):
        try:
            cf = CachedFile.objects.get(key=key)
            df = cf.disk_file
            if not os.path.exists(df.get_path()):
                logmsg("CachedFile's DiskFile '%s' does not exist at path '%s'; deleting database entry."
                       % (str(df), df.get_path()))
                df.delete()
                cf.delete()
                return None
            return df
        except:
            return None

    @staticmethod
    def add(key, filename, collection=DEFAULT_COLLECTION):
        df = DiskFile.from_file(filename, collection)
        cf = CachedFile(disk_file=df, key=key)
        cf.save()
        return df

class Image(models.Model):
    RESIZED_COLLECTION = 'resized'
    ORIG_COLLECTION = 'uploaded'

    MIME_TYPES = {
        'PNG image': 'image/png',
        'JPEG image data': 'image/jpeg',
        'GIF image data': 'image/gif',
        'FITS image data': 'image/jpeg', # fits images are converted to jpg for the browser
    }

    disk_file = models.ForeignKey(DiskFile)
    width = models.PositiveIntegerField(null=True)
    height = models.PositiveIntegerField(null=True)
    thumbnail = models.ForeignKey('Image', related_name='image_thumbnail_set', null=True)
    display_image = models.ForeignKey('Image', related_name='image_display_set', null=True)

    # Reverse mappings:
    #  userimage_set -> UserImage

    #  image_thumbnail_set -> Image
    #  image_display_set -> Image

    def is_source_list(self):
        ''' xy list? '''
        return hasattr(self, 'sourcelist')

    def get_mime_type(self):
        if hasattr(self, 'sourcelist'):
            return self.sourcelist.get_mime_type()
        else:
            return self.MIME_TYPES.get(self.disk_file.file_type, '')

    def get_thumbnail(self):
        if self.thumbnail is None:
            self.thumbnail = self.create_resized_image(256)
            self.save()
        return self.thumbnail

    def get_display_image(self):
        if self.display_image is None:
            self.display_image = self.create_resized_image(640)
            self.save()
        return self.display_image

    def get_image_path(self):
        if hasattr(self, 'sourcelist'):
            return self.sourcelist.get_image_path()
        else:
            return self.disk_file.get_path()

    def get_pnm_path(self):
        imgfn = self.get_image_path()
        #pnmfn = get_temp_file(suffix='.pnm')
        #(filetype, errstr) = image2pnm(imgfn, pnmfn)
        pnmfn = get_temp_file(suffix='.ppm')
        (filetype, errstr) = image2pnm(imgfn, pnmfn, force_ppm=True)
        if errstr:
            raise RuntimeError('Error converting image file %s: %s' % (imgfn, errstr))
        return pnmfn

    def create_resized_image(self, maxsize):
        if max(self.width, self.height) <= maxsize:
            return self
        pnmfn = self.get_pnm_path()
        imagefn = get_temp_file()
        # find scale
        scale = float(maxsize) / float(max(self.width, self.height))
        W,H = int(round(scale * self.width)), int(round(scale * self.height))
        cmd = 'pnmscale -width %i -height %i %s | pnmtojpeg > %s' % (W, H, pnmfn, imagefn)
        logmsg("Making resized image: %s" % cmd)
        rtn,out,err = run_command(cmd)
        if rtn:
            logmsg('pnmscale failed: rtn %i' % rtn)
            logmsg('out: ' + out)
            logmsg('err: ' + err)
            raise RuntimeError('Failed to make resized image for %s: pnmscale: %s' % (str(self), err))
        df = DiskFile.from_file(imagefn, Image.RESIZED_COLLECTION)
        image, created = Image.objects.get_or_create(disk_file=df, width=W, height=H)
        return image

    def render(self, f):
        if hasattr(self, 'sourcelist'):
            # image is a source list
            self.sourcelist.render(f)
        else:
            if self.disk_file.is_fits_image():
                # convert fits image to jpg for browser rendering
                key = 'jpg_image%i' % self.id
                df = CachedFile.get(key)
                if df is None:
                    imagefn = get_temp_file()
                    pnmfn = self.get_pnm_path()
                    cmd = 'pnmtojpeg < %s > %s' % (pnmfn, imagefn)
                    logmsg("Making resized image: %s" % cmd)
                    rtn,out,err = run_command(cmd)
                    if rtn:
                        logmsg('pnmtojpeg failed: rtn %i' % rtn)
                        logmsg('out: ' + out)
                        logmsg('err: ' + err)
                        raise RuntimeError('Failed to make jpg image for %s: pnmtojpeg: %s' % (str(self), err))

                    # cache
                    logmsg('Caching key "%s"' % key)
                    df = CachedFile.add(key, imagefn)
                else:
                    logmsg('Cache hit for key "%s"' % key)
            else:
                df = self.disk_file

            dfile = open(df.get_path())
            f.write(dfile.read())
            dfile.close()

    def get_thumbnail_offset_x(self):
        return (235-self.width)/2

    def get_thumbnail_offset_y(self):
        return (235-self.height)/2

class SourceList(Image):
    SOURCE_TYPE_CHOICES = (('fits','FITS binary table'),
                           ('text','Text list'))

    source_type = models.CharField(max_length=4, choices=SOURCE_TYPE_CHOICES)
    
    def get_fits_path(self):
        if self.source_type == 'fits':
            return self.disk_file.get_path()
        elif self.source_type == 'text':
            key = 'fits_table_df%s' % self.disk_file.file_hash
            df = CachedFile.get(key)
            if df is None:
                fitsfn = get_temp_file()

                text_file = open(str(self.disk_file.get_path()))
                text = text_file.read()
                text_file.close()

                # add x y header
                # potential hack, assumes it doesn't exist...
                text = "# x y\n" + text

                text_table = text_table_fields("", text=text)
                text_table.write_to(fitsfn)
                logmsg("Creating fits table from text list")

                fits = fits_table(fitsfn)

                # cache
                logmsg('Caching key "%s"' % key)
                df = CachedFile.add(key, fitsfn)
            else:
                logmsg('Cache hit for key "%s"' % key)
            return df.get_path()

    def get_fits_table(self):
        table = fits_table(str(self.get_fits_path()))
        return table

    def get_image_path(self):
        imgfn = get_temp_file()
        f = open(imgfn,'wb')
        self.render(f)
        f.close()
        return imgfn

    def get_mime_type(self):
        return 'image/png'

    def create_resized_image(self, maxsize):
        if max(self.width, self.height) <= maxsize:
            return self

        scale = float(maxsize) / float(max(self.width, self.height))
        W,H = int(round(scale * self.width)), int(round(scale * self.height))
        image = SourceList(disk_file=self.disk_file, source_type=self.source_type, width=W, height=H)
        image.save()
        return image
    
    def render(self, f):
        fits = self.get_fits_table()
        #w = int(fits.x.max()-fits.x.min())
        #h = int(fits.y.max()-fits.y.min())
        w = int(np.ceil(fits.x.max()))
        h = int(np.ceil(fits.y.max()))
        scale = float(self.width)/w
        #xmin = int(fits.x.min())
        #ymin = int(fits.y.min())
        xmin = ymin = 1.
        
        img = PIL.Image.new('RGB',(self.width,self.height))
        draw = PIL.ImageDraw.Draw(img)

        r = round(0.001*self.width)
        for (x, y) in zip(fits.x,fits.y):
            x = int((x-xmin)*scale)
            y = int((y-ymin)*scale)
            draw.ellipse((x-r,y-r,x+r+1,y+r+1),fill="rgb(255,255,255)")
        del draw
        img.save(f, 'PNG')
        


class SkyObject(models.Model):
    name = models.CharField(max_length=1024, primary_key=True)


class Tag(models.Model):
    # user = models.ForeignKey(User) # do we need to keep track of who tags what?
    text = models.CharField(max_length=4096, primary_key=True)
    
    # Reverse mappings:
    #  user_images -> UserImage
    #  albums -> Album

class Calibration(models.Model):
    # TAN WCS, straight from the quad match
    raw_tan = models.ForeignKey('TanWCS', related_name='calibrations_raw', null=True)
    # TAN WCS, after tweaking
    tweaked_tan = models.ForeignKey('TanWCS', related_name='calibrations_tweaked', null=True)
    # SIP
    sip = models.ForeignKey('SipWCS', null=True)

    # Reverse mappings:
    #   job  -> Job

    # RA,Dec bounding box.
    ramin  = models.FloatField()
    ramax  = models.FloatField()
    decmin = models.FloatField()
    decmax = models.FloatField()
    
    # cartesian coordinates on unit sphere (for cone search)
    x = models.FloatField()
    y = models.FloatField()
    z = models.FloatField()
    r = models.FloatField()
    
    sky_location = models.ForeignKey('SkyLocation', related_name='calibrations', null=True)

    def __str__(self):
        s = 'Calibration %i' % self.id
        return s

    def get_wcs_file(self):
        return self.job.get_wcs_file()

    def wcs(self):
        return self.raw_tan

    def get_center_radec(self):
        (ra,dec,radius) = self.raw_tan.get_center_radecradius()
        return (ra,dec)

    def get_radius(self):
        return self.raw_tan.get_radius()

    def get_center_radecradius(self):
        return self.raw_tan.get_center_radecradius()

    def format_radec(self):
        r,d = self.get_center_radec()
        return '%.3f, %.3f' % (r, d)

    def format_ra_hms(self):
        r,d = self.get_center_radec()
        h,m,s  = ra2hmsstring(r).split()
        return '%s<sup>h</sup>&nbsp;%s<sup>m</sup>&nbsp;%s<sup>s</sup>' % (h, m, s)

    def format_dec_dms(self):
        r,_d = self.get_center_radec()
        d,m,s = dec2dmsstring(_d).split()
        return '%s&deg;&nbsp;%s\'&nbsp;%s"' % (d, m, s)

    def format_radius(self):
        ## FIXME -- choose units and resolution (deg, arcmin)
        r,d,radius = self.get_center_radecradius()
        return '%.3f deg' % radius

    def format_size(self):
        w,h,units = self.raw_tan.get_field_size()
        return '%.3g x %.3g %s' % (w, h, units)

    def format_pixscale(self):
        s = self.raw_tan.get_pixscale()
        return '%.3g arcsec/pixel' % s

    def format_orientation(self):
        o = self.raw_tan.get_orientation()
        return 'Up is %.3g degrees E of N' % o
    
    def get_objs_in_field(self):
        def run_convert_command(cmd, deleteonfail=None):
            logmsg('Command: ' + cmd)
            (rtn, stdout, stderr) = run_command(cmd)
            if rtn:
                errmsg = 'Command failed: ' + cmd + ': ' + stderr
                logmsg(errmsg + '; rtn val %d' % rtn)
                logmsg('out: ' + stdout);
                logmsg('err: ' + stderr);
                if deleteonfail:
                    os.unlink(deleteonfail)
                raise FileConversionError(errmsg)

        def annotate_command(job):
            hd = False
            wcs = job.calibration.tweaked_tan
            if wcs:
                # one square degree
                hd = (wcs.get_field_area() < 1.)
            wcsfn = job.get_wcs_file()

            #cmd = 'plotann.py %s' % wcsfn
            
            cmd = 'plot-constellations -w %s -N -C -B -b 10 -j' % wcsfn
            if hd:
                cmd += ' -D -d %s' % settings.HENRY_DRAPER_CAT
            return cmd
        
        objs = []
        cmd = annotate_command(self.job)
        cmd += '-L > %s' % self.job.get_obj_file()
        run_convert_command(cmd)
        objfile = open(self.job.get_obj_file(), 'r')
        objtxt = objfile.read()
        objfile.close()
        for objline in objtxt.split('\n'):
            for obj in objline.split('/'):
                obj = obj.strip()
                if obj != '':
                    objs.append(obj)
        return objs


class Job(models.Model):
    calibration = models.OneToOneField('Calibration', null=True,
        related_name="job")
    
    STATUS_CHOICES = (
        ('S', 'Success'), 
        ('F', 'Failure'),
    )    
    
    status = models.CharField(max_length=1, choices=STATUS_CHOICES)
    error_message = models.CharField(max_length=256)
    user_image = models.ForeignKey('UserImage', related_name='jobs')

    queued_time = models.DateTimeField(null=True)
    start_time = models.DateTimeField(null=True)
    end_time = models.DateTimeField(null=True)

    # Reverse mappings:
    #  none

    def __str__(self):
        s = 'Job %i' % self.id
        if self.calibration is not None:
            s += ', calib %i' % self.calibration.id
        if self.end_time is not None:
            s += ', end time ' + str(self.end_time)
        return s

    def set_queued_time(self):
        self.queued_time = datetime.now()

    def set_start_time(self):
        self.start_time = datetime.now()

    def set_end_time(self):
        self.end_time = datetime.now()

    def OLD_get_dir(self):
        if self.id < 10000:
            return os.path.join('old-jobs-10000', '%08i' % self.id)
        return os.path.join('old-jobs', '%08i' % self.id)

    def get_dir(self):
        jtxt = '%08i' % self.id
        return os.path.join(JOBDIR, jtxt[:4], jtxt)

    def get_axy_file(self):
        return os.path.join(self.get_dir(), 'job.axy')
        
    def get_wcs_file(self):
        return os.path.join(self.get_dir(), 'wcs.fits')

    def get_rdls_file(self):
        return os.path.join(self.get_dir(), 'rdls.fits')

    def get_obj_file(self):
        return os.path.join(self.get_dir(), 'objsinfield')

    def get_log_file(self):
        return os.path.join(self.get_dir(), 'log')

    def get_log_tail(self, nlines=20):
        fn = self.get_log_file()
        if not os.path.exists(fn):
            return None
        if nlines is None:
            return open(fn).read()
        lines = open(fn).readlines()
        return ''.join(lines[-nlines:])

    # HACK
    def get_log_file2(self):
        return os.path.join(self.get_dir(), 'job.log')

    # HACK
    def get_log_tail2(self, nlines=20):
        fn = self.get_log_file2()
        if not os.path.exists(fn):
            return None
        if nlines is None:
            return open(fn).read()
        lines = open(fn).readlines()
        return ''.join(lines[-nlines:])

    def make_dir(self):
        dirnm = self.get_dir()
        # remove any previous contents
        shutil.rmtree(dirnm, ignore_errors=True)
        if not os.path.exists(dirnm):
            os.makedirs(dirnm)
        return dirnm

    def get_status_blurb(self):
        blurb = "processing"
        if self.start_time:
            if not self.end_time:
                blurb = "solving"
            else:
                if self.status == 'S':
                    blurb = "success"
                elif self.status == 'F':
                    blurb = "failure"
                else:
                    blurb = '?'
        return blurb


class SkyLocation(models.Model):
    nside = models.PositiveSmallIntegerField()
    healpix = models.BigIntegerField()

    # Reverse mappings:
    #  calibrations -> Calibration

    def __str__(self):
        s = '<SkyLocation: nside(%i) healpix(%i)>' % (self.nside, self.healpix)
        return s
    
    def get_user_images(self, nside=None, healpix=None):
        # NOTE: this returns a queryset
        if nside is None or healpix is None:
            nside = self.nside
            healpix = self.healpix
        user_images = UserImage.objects.all_visible()
        user_images = user_images.filter(jobs__calibration__sky_location__nside=nside)
        user_images = user_images.filter(jobs__calibration__sky_location__healpix=healpix)
        return user_images
        
    def get_neighbouring_user_images(self):
        user_images = self.get_user_images()
        # add neighbors at current scale
        neighbours = anutil.healpix_get_neighbours(self.healpix, self.nside)
        for hp in neighbours:
            user_images |= self.get_user_images(self.nside, hp)

        # next bigger scale
        user_images |= self.get_user_images(self.nside/2, self.healpix/4)

        # next smaller scale
        neighbours = set()
        for i in range(4):
            n = anutil.healpix_get_neighbours(self.healpix*4+i, self.nside*2)
            neighbours.update(n)
        for hp in neighbours:
            user_images |= self.get_user_images(self.nside*2, hp)
        
        return user_images

class FlaggedUserImage(models.Model):
    user_image = models.ForeignKey('UserImage')
    flag = models.ForeignKey('Flag')
    user = models.ForeignKey(User)
    flagged_time = models.DateTimeField(auto_now=True)

class TaggedUserImage(models.Model):
    user_image = models.ForeignKey('UserImage')
    tag = models.ForeignKey('Tag')
    tagger = models.ForeignKey(User, null=True)
    added_time = models.DateTimeField(auto_now=True) 


class UserImageManager(models.Manager):
    def admin_all(self):
        return super(UserImageManager, self)

    def all_visible(self):
        anonymous = User.objects.get(username=ANONYMOUS_USERNAME)
        solved_anonymous_uis = super(UserImageManager, self).filter(user=anonymous, jobs__calibration__isnull=False)
        non_anonymous_uis = super(UserImageManager, self).exclude(user=anonymous)
        valid_uis = non_anonymous_uis | solved_anonymous_uis
        return valid_uis.order_by('-submission__submitted_on')

    def public_only(self, user=None):
        if user and not user.is_authenticated():
            user = None
        return self.all_visible().filter(Q(publicly_visible='y') | Q(user=user))
    

class UserImage(Hideable):
    objects = UserImageManager()

    image = models.ForeignKey('Image')
    user = models.ForeignKey(User, related_name='user_images', null=True)
    
    tags = models.ManyToManyField('Tag',related_name='user_images',
        through='TaggedUserImage')

    flags = models.ManyToManyField('Flag', related_name='user_images',
        through='FlaggedUserImage')

    sky_objects = models.ManyToManyField('SkyObject', related_name='user_images')

    description = models.CharField(max_length=1024, blank=True)
    original_file_name = models.CharField(max_length=256)
    submission = models.ForeignKey('Submission', related_name='user_images')

    license = models.ForeignKey('License')
    comment_receiver = models.OneToOneField('CommentReceiver')

    # Reverse mappings:
    #  jobs -> Job
    #  albums -> Album

    def save(self, *args, **kwargs):
        self.owner = self.user
        #self.license.save(self.user.get_profile().default_license)
        return super(UserImage, self).save(*args, **kwargs)


    def add_sky_objects(self, job):
        logmsg('adding sky objects for %s' % self)
        sky_objects = job.calibration.get_objs_in_field()
        for sky_object in sky_objects:
            log_tag = unicode(sky_object,errors='ignore')
            logmsg(u'getting or creating sky object %s' % log_tag)
            sky_obj,created = SkyObject.objects.get_or_create(name=sky_object)
            if created:
                logmsg('created sky objects')
            self.sky_objects.add(sky_obj)
        logmsg('done adding machine tags')
        #self.save()


    def add_machine_tags(self, job):
        logmsg('adding machine tags for %s' % self)
        sky_objects = job.calibration.get_objs_in_field()
        for sky_object in sky_objects:
            log_tag = unicode(sky_object,errors='ignore')
            logmsg(u'getting or creating machine tag %s' % log_tag)
            machine_tag,created = Tag.objects.get_or_create(text=sky_object)
            if created:
                logmsg('created machine tag')

            # associate this UserImage with the machine tag
            logmsg(u'adding machine tag: %s' % log_tag)
            machine_user = User.objects.get(username=MACHINE_USERNAME)
            tagged_user_image = TaggedUserImage.objects.get_or_create(
                user_image=self,
                tag=machine_tag,
                tagger=machine_user,
            )
            logmsg('tagged user image saved')
        logmsg('done adding machine tags')
        #self.save()
                
    def get_best_job(self):
        jobs = self.jobs.all()
        if jobs.count() == 1:
            return jobs[0]
        # Keep latest solved
        j1 = jobs.filter(status='S')
        # ?
        j1 = jobs.filter(calibration__isnull=False)
        if j1.count():
            jobs = j1
        # FIXME
        jobs = jobs.order_by('-end_time')
        if len(jobs):
            return jobs[0]
        return None

    def get_absolute_url(self):
        kwargs = {'user_image_id':self.id}
        abs_url = reverse('astrometry.net.views.image.user_image', kwargs=kwargs)
        return abs_url
    
    def is_calibrated(self):
        job = self.get_best_job()
        return (job and job.calibration)

    def get_neighbouring_user_images(self):
        if self.is_calibrated():
            job = self.get_best_job()
            images = job.calibration.sky_location.get_neighbouring_user_images()
            images = images.exclude(pk=self.id)
        else:
            images = UserImage.objects.none()
        return images

    def update_flags(self, selected_flag_names, flagger):
        flags = Flag.objects.all()
        for flag in flags:
            if flag.name in selected_flag_names:
                logmsg('flagging ui %d: %s' % (self.pk, flag.name))
                FlaggedUserImage.objects.get_or_create(
                    user_image=self,
                    flag=flag,
                    user=flagger,
                )
            else:
                try:
                    logmsg('removing flag %s from ui %d' % (flag.name, self.pk))
                    FlaggedUserImage.objects.filter(
                        user_image=self,
                        flag=flag,
                        user=flagger,
                    ).delete()
                except ObjectDoesNotExist:
                    pass


class Submission(Hideable):
    SCALEUNITS_CHOICES = (
        ('arcsecperpix', 'arcseconds per pixel'),
        ('arcminwidth' , 'width of the field (in arcminutes)'), 
        ('degwidth' , 'width of the field (in degrees)'),
        ('focalmm'     , 'focal length of the lens (for 35mm film equivalent sensor)'),
    )
    scaleunits_default = 'degwidth'

    SCALETYPE_CHOICES = (
        ('ul', 'bounds'),
        ('ev', 'estimate +/- error'),
    )

    PARITY_CHOICES = (
        (2, 'try both simultaneously'),
        (0, 'positive'),
        (1, 'negative'),
    )

    '''SOURCE_TYPE_CHOICES = (
        ('image', 'image'),
        ('fits', 'FITS binary table'),
        ('text', 'text list'),
    )'''
    ###
    user = models.ForeignKey(User, related_name='submissions', null=True)
    disk_file = models.ForeignKey(DiskFile, related_name='submissions', null=True)
    url = models.URLField(blank=True, null=True)
    parity = models.PositiveSmallIntegerField(choices=PARITY_CHOICES, default=2)
    scale_units = models.CharField(max_length=20, choices=SCALEUNITS_CHOICES, default='degwidth')
    scale_type = models.CharField(max_length=2, choices=SCALETYPE_CHOICES, default='ul')
    scale_lower = models.FloatField(default=0.1, blank=True, null=True)
    scale_upper = models.FloatField(default=180, blank=True, null=True)
    scale_est   = models.FloatField(blank=True, null=True)
    scale_err   = models.FloatField(blank=True, null=True)
    
    positional_error = models.FloatField(blank=True, null=True)
    center_ra = models.FloatField(blank=True, null=True)
    center_dec = models.FloatField(blank=True, null=True)
    radius = models.FloatField(blank=True, null=True)

    tweak_order = models.IntegerField(blank=True, null=True, default=2)

    downsample_factor = models.PositiveIntegerField(blank=True, null=True, default=2)

    use_sextractor = models.BooleanField(default=False)
    crpix_center = models.BooleanField(default=False)

    #source_type = models.CharField(max_length=5, choices=SOURCE_TYPE_CHOICES, default='image')
    original_filename = models.CharField(max_length=256)
    album = models.ForeignKey('Album', blank=True, null=True)

    submitted_on = models.DateTimeField(auto_now_add=True)
    # This field is used as a marker that the job has been submitted to the
    # worker pool for processing.  ACTUAL processing may not happen until
    # later.
    processing_started = models.DateTimeField(null=True)
    processing_finished = models.DateTimeField(null=True)

    processing_retries = models.PositiveIntegerField(default=0)

    error_message = models.CharField(max_length=2048, null=True)

    license = models.ForeignKey('License')
    comment_receiver = models.OneToOneField('CommentReceiver')

    # Reverse mappings:
    #  user_images -> UserImage
    #  -> QueuedSubmission

    def __str__(self):
        return ('Submission %i: file <%s>, url %s, proc_started=%s' %
                (self.id, str(self.disk_file), self.url, str(self.processing_started)))

    def set_error_message(self, msg):
        if len(msg) > 255:
            msg = '...' + msg[-252:]
        self.error_message = msg

    def get_absolute_url(self):
        kwargs = {'subid':self.id}
        abs_url = reverse('astrometry.net.views.submission.status', kwargs=kwargs)
        return abs_url

    def get_user_image(self):
        uis = self.user_images.all()
        if uis.count():
            return uis[0]
        return None

    def get_best_jobs(self):
        uimgs = self.user_images.all()
        return [u.get_best_job() for u in uimgs]

    def get_scale_bounds(self):
        stype = self.scale_type
        if stype == 'ul':
            return (self.scale_lower, self.scale_upper)
        elif stype == 'ev':
            est = self.scale_est
            err = self.scale_err
            return (est * (1.0 - err / 100.0),
                    est * (1.0 + err / 100.0))
        else:
            return None

    def set_processing_started(self):
        self.processing_started = datetime.now()
    def set_processing_finished(self):
        self.processing_finished = datetime.now()

    def save(self, *args, **kwargs):
        default_license=self.user.get_profile().default_license
        try:
            self.license
        except:
            self.license = License.objects.create(
                allow_modifications=default.allow_modifications,
                allow_commercial_use=default.allow_commercial_use,
            )
        try:
            self.comment_receiver
        except:
            self.comment_receiver = CommentReceiver.objects.create()

        self.comment_receiver.save()
        #self.license.save(default_license=default_license)
            
        logmsg('saving submission: license id = %d' % self.license.id)
        logmsg('saving submission: commentreceiver id = %d' % self.comment_receiver.id)

        now = datetime.now()
        return super(Submission, self).save(*args, **kwargs)


class Album(Hideable):
    user = models.ForeignKey(User, related_name='albums', null=True)
    title = models.CharField(max_length=64)
    description = models.CharField(max_length=1024, blank=True)
    user_images = models.ManyToManyField('UserImage', related_name='albums') 
    tags = models.ManyToManyField('Tag', related_name='albums')
    created_at = models.DateTimeField(auto_now_add=True)

    comment_receiver = models.OneToOneField('CommentReceiver')

    def save(self, *args, **kwargs):
        self.owner = self.user
        return super(Album, self).save(*args, **kwargs)

    def get_absolute_url(self):
        kwargs = {'album_id':self.id}
        abs_url = reverse('astrometry.net.views.album.album', kwargs=kwargs)
        return abs_url
        
class Comment(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    recipient = models.ForeignKey('CommentReceiver', related_name='comments')
    author = models.ForeignKey(User, related_name='comments_left')
    text = models.CharField(max_length=1024)

    class Meta:
        ordering = ["-created_at"]


class UserProfile(models.Model):
    API_KEY_LENGTH = 16
    display_name = models.CharField(max_length=32)
    user = models.ForeignKey(User, unique=True, related_name='profile',
                             editable=False)
    apikey = models.CharField(max_length = API_KEY_LENGTH)
    default_license = models.ForeignKey('License', default=DEFAULT_LICENSE_ID)

    def __str__(self):
        s = ('UserProfile: user %s, API key %s' % (self.user.get_full_name().encode('ascii','replace'), self.apikey))
        return s

    def create_api_key(self):
        # called in openid_views.py (where profile is first created)
        key = ''.join([chr(random.randint(ord('a'), ord('z')))
                       for i in range(self.__class__.API_KEY_LENGTH)])
        self.apikey = key
     
    def create_default_license(self):
        # make a user their own copy of the sitewide default license
        # called in openid_views.py (where profile is first created)
        sdl = License.get_default()
        if self.default_license == None or self.default_license.id == sdl.id:
            self.default_license = License.objects.create(
                allow_modifications=sdl.allow_modifications,
                allow_commercial_use=sdl.allow_commercial_use
            )

    def get_absolute_url(self):
        return reverse('astrometry.net.views.user.public_profile', user_id=self.user.id)

    def save(self, *args, **kwargs):
        # for sorting users, enforce capitalization of first letter
        self.display_name = self.display_name[:1].capitalize() + self.display_name[1:]

        return super(UserProfile, self).save(*args, **kwargs)
