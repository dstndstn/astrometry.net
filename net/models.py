import random
import os
import errno
import hashlib
import shutil
import tempfile
from datetime import datetime
from copy import deepcopy

from django.db import models
from django.contrib.auth.models import User

from django.core.exceptions import ObjectDoesNotExist
from django.core.urlresolvers import reverse

from astrometry.net.settings import *
from wcs import *
from log import *

from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command
from astrometry.util.pyfits_utils import *
from astrometry.net.tmpfile import *

import PIL.Image, PIL.ImageDraw

from urllib2 import urlopen

from astrometry.net.abstract_models import *

### Admin view -- running Submissions and Jobs

# = dt.total_seconds() in python 2.7
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

class License(Licensable):
    def save(self, *args, **kwargs):
        self.get_license_name_uri()
        return super(License, self).save(*args,**kwargs)

    @staticmethod
    def get_default():
        # DEFAULT_LICENSE_ID  defined in settings_common.py
        return License.objects.get(pk=DEFAULT_LICENSE_ID)



class DiskFile(models.Model):
    file_hash = models.CharField(max_length=40, unique=True, primary_key=True)
    size = models.PositiveIntegerField()
    file_type = models.CharField(max_length=256, null=True)

    # Reverse mappings:
    #  image_set -> Image
    #  submissions -> Submission

    def __str__(self):
        return 'DiskFile: %s, size %i, type %s' % (self.file_hash, self.size, self.file_type)

    def set_size_and_file_type(self):
        fn = self.get_path()
        st = os.stat(fn)
        self.size = st.st_size
        filetypes = filetype_short(fn)
        self.file_type = ';'.join(filetypes)

    def get_file_types(self):
        return self.file_type.split(';')

    def get_path(self):
        return DiskFile.get_file_path(self.file_hash)

    @staticmethod
    def get_file_directory(file_hash_digest):
        return os.path.join(DATADIR,
                            file_hash_digest[0:2],
                            file_hash_digest[2:4],
                            file_hash_digest[4:6])

    @staticmethod
    def get_file_path(file_hash_digest):
        file_path = DiskFile.get_file_directory(file_hash_digest)
        file_path = os.path.join(file_path, file_hash_digest)
        return file_path

    @staticmethod
    def make_dirs(file_hash_digest):
        file_directory = DiskFile.get_file_directory(file_hash_digest)
        try:
            os.makedirs(file_directory)
        except OSError as e:
            # we don't care if the directory already exists
            if e.errno == errno.EEXIST:
                pass
            else: raise

    @staticmethod
    def from_file(filename):
        file_hash = DiskFile.get_hash()
        f = open(filename)
        while True:
            s = f.read(8096)
            if not len(s):
                # EOF
                break
            file_hash.update(s)
        hashkey = file_hash.hexdigest()
        df,created = DiskFile.objects.get_or_create(file_hash=hashkey,
                                                    defaults=dict(size=0, file_type=''))
        if created:
            # move it into place
            DiskFile.make_dirs(hashkey)
            shutil.move(filename, DiskFile.get_file_path(hashkey))
            df.set_size_and_file_type()
            df.save()
        return df

    @staticmethod
    def get_hash():
        return hashlib.sha1()

class CachedFile(models.Model):
    disk_file = models.ForeignKey(DiskFile)
    key = models.CharField(max_length=64, unique=True, primary_key=True)

    @staticmethod
    def get(key):
        try:
            cf = CachedFile.objects.get(key=key)
            return cf.disk_file
        except:
            return None

    @staticmethod
    def add(key, filename):
        df = DiskFile.from_file(filename)
        cf = CachedFile(disk_file=df, key=key)
        cf.save()
        return df

class Image(models.Model):
    disk_file = models.ForeignKey(DiskFile)
    width = models.PositiveIntegerField(null=True)
    height = models.PositiveIntegerField(null=True)
    thumbnail = models.ForeignKey('Image', related_name='image_thumbnail_set', null=True)
    display_image = models.ForeignKey('Image', related_name='image_display_set', null=True)

    # Reverse mappings:
    #  userimage_set -> UserImage

    def get_mime_type(self):
        return self.disk_file.file_type

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

    def create_resized_image(self, maxsize):
        if max(self.width, self.height) <= maxsize:
            return self
        from astrometry.util.image2pnm import image2pnm
        fn = self.disk_file.get_path()
        f,tmpfn = tempfile.mkstemp()
        os.close(f)
        (ext,err) = image2pnm(fn, tmpfn)
        if ext is None:
            raise RuntimeError('Failed to make resized image for %s: image2pnm: %s' % (str(self), err))
        f,imagefn = tempfile.mkstemp()
        os.close(f)
        # find scale
        scale = float(maxsize) / float(max(self.width, self.height))
        W,H = int(round(scale * self.width)), int(round(scale * self.height))
        cmd = 'pnmscale -width %i -height %i %s | pnmtojpeg > %s' % (W, H, tmpfn, imagefn)
        logmsg("Making resized image: %s" % cmd)
        rtn,out,err = run_command(cmd)
        if rtn:
            logmsg('pnmscale failed: rtn %i' % rtn)
            logmsg('out: ' + out)
            logmsg('err: ' + err)
            raise RuntimeError('Failed to make resized image for %s: pnmscale: %s' % (str(self), err))
        df = DiskFile.from_file(imagefn)
        image = Image(disk_file=df, width=W, height=H)
        image.save()
        return image

    def render(self, f):
        if hasattr(self,'sourcelist'):
            # image is a source list
            self.sourcelist.render(f)
        else:
            df = open(self.disk_file.get_path())
            f.write(df.read())
            df.close()

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
                cmd = 'text2fits.py %s %s' % (self.disk_file.get_path(), fitsfn)
                logmsg("Creating fits table from text list: %s" % cmd)
                rtn,out,err = run_command(cmd)
                if rtn:
                    logmsg('text2fits.py failed: rtn %i' % rtn)
                    logmsg('out: ' + out)
                    logmsg('err: ' + err)
                    raise RuntimeError('Failed to create fits table from %s: text2fits.py: %s' % (str(self), err))

                # cache
                logmsg('Caching key "%s"' % key)
                df = CachedFile.add(key, fitsfn)
            else:
                logmsg('Cache hit for key "%s"' % key)
            return df.get_path()

    def get_fits_table(self):
        table = fits_table(str(self.get_fits_path()))
        return table

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
        w = fits.x.max()-fits.x.min()
        h = fits.y.max()-fits.y.min()
        scale = float(self.width)/(1.01*w) 
        xmin = fits.x.min()-0.005*w
        ymin = fits.y.min()-0.005*h
        
        img = PIL.Image.new('RGB',(self.width,self.height))
        draw = PIL.ImageDraw.Draw(img)

        r = round(0.001*self.width)
        for (x, y) in zip(fits.x,fits.y):
            x = int((x-xmin)*scale)
            y = int((y-ymin)*scale)
            draw.ellipse((x-r,y-r,x+r+1,y+r+1),fill="rgb(255,255,255)")
        del draw
        img.save(f, 'PNG')
        


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
    #   jobs  -> Job

    # RA,Dec bounding box.
    ramin  = models.FloatField()
    ramax  = models.FloatField()
    decmin = models.FloatField()
    decmax = models.FloatField()

    def __str__(self):
        s = 'Calibration %i' % self.id
        return s

    def get_wcs_file(self):
        jobs = self.jobs.all()
        if len(jobs) == 0:
            logmsg('Calibration.wcs_path: I have no Jobs: my id=%i' % self.id)
            return None
        job = jobs[0]
        logmsg('Calibration: job is', job)
        return job.get_wcs_file()

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

    def format_radius(self):
        ## FIXME -- choose units and resolution (deg, arcmin)
        r,d,radius = self.get_center_radecradius()
        return '%.3f deg' % radius

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
            cmd = 'plot-constellations -w %s -N -C -B -b 10 -j' % wcsfn
            if hd:
                cmd += ' -D -d %s' % settings.HENRY_DRAPER_CAT
            return cmd
        
        objs = []
        for job in self.jobs.all():
            cmd = annotate_command(job)
            cmd += '-L > %s' % job.get_obj_file()
            run_convert_command(cmd)
            objfile = open(job.get_obj_file(), 'r')
            objtxt = objfile.read()
            objfile.close()
            for objline in objtxt.split('\n'):
                for obj in objline.split('/'):
                    obj = obj.strip()
                    if obj != '':
                        objs.append(obj)
        return objs


class Job(models.Model):
    calibration = models.ForeignKey('Calibration', null=True,
        related_name="jobs")
    
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

    def get_dir(self):
        return os.path.join(JOBDIR, '%08i' % self.id)

    def get_wcs_file(self):
        return os.path.join(self.get_dir(), 'wcs.fits')

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

class TaggedUserImage(models.Model):
    user_image = models.ForeignKey('UserImage')
    tag = models.ForeignKey('Tag')
    tagger = models.ForeignKey(User, null=True)
    added_time = models.DateTimeField(auto_now=True) 


class UserImageManager(models.Manager):
    def admin_all(self):
        return super(UserImageManager, self)

    def all(self):
        anonymous = User.objects.get(username=ANONYMOUS_USERNAME)
        non_anonymous_uis = super(UserImageManager, self).filter(user=anonymous, jobs__calibration__isnull=False)
        solved_anonymous_uis = super(UserImageManager, self).exclude(user=anonymous)
        valid_uis = non_anonymous_uis | solved_anonymous_uis
        return valid_uis.order_by('-submission__submitted_on')


class UserImage(Commentable, Licensable, Hideable):
    objects = UserImageManager()

    image = models.ForeignKey('Image')
    user = models.ForeignKey(User, related_name='user_images', null=True)
    
    tags = models.ManyToManyField('Tag',related_name='user_images',
        through='TaggedUserImage')

    description = models.CharField(max_length=1024)
    original_file_name = models.CharField(max_length=256)
    submission = models.ForeignKey('Submission', related_name='user_images')

    # Reverse mappings:
    #  jobs -> Job
    #  albums -> Album

    def save(self, *args, **kwargs):
        self.owner = self.user
        self.get_license_name_uri()
        return super(UserImage, self).save(*args, **kwargs)


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


class Submission(Licensable, Hideable):
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

    SOURCE_TYPE_CHOICES = (
        ('image', 'image'),
        ('fits', 'FITS binary table'),
        ('text', 'text list'),
    )
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
    downsample_factor = models.PositiveIntegerField(blank=True, null=True)

    source_type = models.CharField(max_length=5, choices=SOURCE_TYPE_CHOICES, default='image')
    original_filename = models.CharField(max_length=256)
    album = models.ForeignKey('Album', blank=True, null=True)

    submitted_on = models.DateTimeField(auto_now_add=True)
    processing_started = models.DateTimeField(null=True)
    processing_finished = models.DateTimeField(null=True)
    error_message = models.CharField(max_length=256, null=True)

    # Reverse mappings:
    #  user_images -> UserImage
    #  -> QueuedSubmission

    def __str__(self):
        return ('Submission %i: file <%s>, url %s, proc_started=%s' %
                (self.id, str(self.disk_file), self.url, str(self.processing_started)))

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


class Album(Commentable, Hideable):
    user = models.ForeignKey(User, related_name='albums', null=True)
    title = models.CharField(max_length=64)
    description = models.CharField(max_length=1024, blank=True)
    user_images = models.ManyToManyField('UserImage', related_name='albums') 
    tags = models.ManyToManyField('Tag', related_name='albums')
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        self.owner = self.user
        return super(Album, self).save(*args, **kwargs)

    def get_absolute_url(self):
        kwargs = {'album_id':self.id}
        abs_url = reverse('astrometry.net.views.album.album', kwargs=kwargs)
        return abs_url
        
class Comment(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    recipient = models.ForeignKey('Commentable', related_name='comments')
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
        s = ('UserProfile: user %s, API key %s' % (self.user.get_full_name(), self.apikey))
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

        self.default_license.save()
        return super(UserProfile, self).save(*args, **kwargs)
