import os
import errno
import hashlib
import shutil
import tempfile
from datetime import datetime

from django.db import models
from django.contrib.auth.models import User

from astrometry.net.settings import *
from userprofile import UserProfile
from wcs import *
from log import *

from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command

class Commentable(models.Model):
    id = models.AutoField(primary_key=True)
    owner = models.ForeignKey(User, null=True)

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
        file_path = os.path.join(DATADIR, file_path, file_hash_digest)
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
        return self.thumbnail

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


class Tag(models.Model):
    user = models.ForeignKey(User)
    text = models.CharField(max_length=4096)
    added_time = models.DateTimeField(auto_now=True) 

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
    #   job_set  -> Job

    # RA,Dec bounding box.
    ramin  = models.FloatField()
    ramax  = models.FloatField()
    decmin = models.FloatField()
    decmax = models.FloatField()

    def __str__(self):
        s = 'Calibration %i' % self.id
        return s

    def get_wcs_file(self):
        jobs = self.job_set.all()
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

    def get_center_radecradius(self):
        return self.raw_tan.get_center_radecradius()

    def format_radec(self):
        r,d = self.get_center_radec()
        return '%.3f, %.3f' % (r, d)

    def format_radius(self):
        ## FIXME -- choose units and resolution (deg, arcmin)
        r,d,radius = self.get_center_radecradius()
        return '%.3f deg' % radius


class Job(models.Model):
    calibration = models.ForeignKey('Calibration', null=True)
    
    STATUS_CHOICES = (
        ('S', 'Success'), 
        ('F', 'Failure'),
    )    
    
    status = models.CharField(max_length=1, choices=STATUS_CHOICES)
    error_message = models.CharField(max_length=256)
    user_image = models.ForeignKey('UserImage', related_name='jobs')

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

    def set_start_time(self):
        self.start_time = datetime.now()

    def set_end_time(self):
        self.end_time = datetime.now()

    def get_dir(self):
        return os.path.join(JOBDIR, '%08i' % self.id)

    def get_wcs_file(self):
        return os.path.join(self.get_dir(), 'wcs.fits')

    def make_dir(self):
        dirnm = self.get_dir()
        if not os.path.exists(dirnm):
            os.makedirs(dirnm)
        return dirnm


class UserImage(Commentable):
    image = models.ForeignKey('Image')
    user = models.ForeignKey(User, related_name='user_images', null=True)
    
    PERMISSION_CHOICES = (('pu', 'Public'),
             ('pr', 'Private'))

    permission = models.CharField(max_length=2, choices=PERMISSION_CHOICES)
    tags = models.ManyToManyField('Tag', related_name='user_images')
    description = models.CharField(max_length=1024)
    original_file_name = models.CharField(max_length=256)
    submission = models.ForeignKey('Submission', related_name='user_images')

    # Reverse mappings:
    #  jobs -> Job
    #  albums -> Album

    def save(self, *args, **kwargs):
        self.owner = self.user
        return super(UserImage, self).save(*args, **kwargs)

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
        
class Submission(models.Model):
    SCALEUNITS_CHOICES = (
        ('arcsecperpix', 'arcseconds per pixel'),
        ('arcminwidth' , 'width of the field (in arcminutes)'), 
        ('degwidth' , 'width of the field (in degrees)'),
        ('focalmm'     , 'focal length of the lens (for 35mm film equivalent sensor)'),
    )
    scaleunits_default = 'degwidth'

    SCALETYPE_CHOICES = (
        ('ul', 'lower and upper bounds'),
        ('ev', 'estimate and error bound'),
    )

    PARITY_CHOICES = (
        (2, 'Try both simultaneously'),
        (0, 'Positive'),
        (1, 'Negative'),
    )

    ###
    user = models.ForeignKey(User, related_name='submissions', null=True)
    disk_file = models.ForeignKey(DiskFile, related_name='submissions', null=True)
    url = models.URLField(blank=True, null=True)
    parity = models.PositiveSmallIntegerField(choices=PARITY_CHOICES, default=2)
    scale_units = models.CharField(max_length=20, choices=SCALEUNITS_CHOICES)
    scale_type = models.CharField(max_length=2, choices=SCALETYPE_CHOICES)
    scale_lower = models.FloatField(default=0.1, blank=True, null=True)
    scale_upper = models.FloatField(default=180, blank=True, null=True)
    scale_est   = models.FloatField(blank=True, null=True)
    scale_err   = models.FloatField(blank=True, null=True)

    original_filename = models.CharField(max_length=256)

    processing_started = models.DateTimeField(null=True)
    processing_finished = models.DateTimeField(null=True)

    # Reverse mappings:
    #  user_images -> UserImage

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


class Album(Commentable):
    description = models.CharField(max_length=1024)
    user_images = models.ManyToManyField('UserImage', related_name='albums') 
    tags = models.ManyToManyField('Tag', related_name='albums')

class Comment(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    recipient = models.ForeignKey('Commentable', related_name='comments')
    author = models.ForeignKey(User, related_name='comments_left')
    text = models.CharField(max_length=1024)

    class Meta:
        ordering = ["-created_at"]
