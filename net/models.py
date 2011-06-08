import os
import errno
from astrometry.net.settings import *
from django.db import models
from django.contrib.auth.models import User
from userprofile import UserProfile
from wcs import *
from datetime import datetime
import hashlib
import shutil
from astrometry.util.filetype import filetype_short

class DiskFile(models.Model):
    file_hash = models.CharField(max_length=40, unique=True, primary_key=True)
    size = models.PositiveIntegerField()
    file_type = models.CharField(max_length=256, null=True)

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

class Tag(models.Model):
    user = models.ForeignKey(User)
    text = models.CharField(max_length=4096)
    added_time = models.DateTimeField(auto_now=True) 

class Calibration(models.Model):
    # TAN WCS, straight from the quad match
    raw_tan = models.ForeignKey('TanWCS', related_name='calibrations_raw', null=True)
    # TAN WCS, after tweaking
    tweaked_tan = models.ForeignKey('TanWCS', related_name='calibrations_tweaked', null=True)
    # SIP
    sip = models.ForeignKey('SipWCS', null=True)

    # RA,Dec bounding box.
    ramin  = models.FloatField()
    ramax  = models.FloatField()
    decmin = models.FloatField()
    decmax = models.FloatField()

    def __str__(self):
        s = 'Calibration %i' % self.id
        return s

class Job(models.Model):
    calibration = models.ForeignKey('Calibration', null=True)
    
    STATUS_CHOICES = (('S', 'Success'), 
              ('F', 'Failure'))    
    
    status = models.CharField(max_length=1, choices=STATUS_CHOICES)
    error_message = models.CharField(max_length=256)
    user_image = models.ForeignKey('UserImage')

    start_time = models.DateTimeField(null=True)
    end_time = models.DateTimeField(null=True)

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

    def make_dir(self):
        dirnm = self.get_dir()
        if not os.path.exists(dirnm):
            os.makedirs(dirnm)
        return dirnm


class UserImage(models.Model):
    image = models.ForeignKey(Image)
    
    PERMISSION_CHOICES = (('pu', 'Public'),
             ('pr', 'Private'))

    permission = models.CharField(max_length=2, choices=PERMISSION_CHOICES)
    tags = models.ManyToManyField('Tag', related_name='images')
    description = models.CharField(max_length=1024)
    original_file_name = models.CharField(max_length=256)
    submission = models.ForeignKey('Submission', related_name='user_images')

    def get_best_job(self):
        jobs = self.job_set.all()
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
        return jobs[0]
        
        
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
    disk_file = models.ForeignKey(DiskFile, related_name='submissions')
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

    def __str__(self):
        return ('Submission %i: file <%s>, url %s, proc_started=%s' %
                (self.id, str(self.disk_file), self.url, str(self.processing_started)))

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


class Album(models.Model):
    description = models.CharField(max_length=1024)
    user_images = models.ManyToManyField('UserImage', related_name='albums') 
    tags = models.ManyToManyField('Tag', related_name='albums') 
