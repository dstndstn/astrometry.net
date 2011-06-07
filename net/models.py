from astrometry.net.settings import *
from django.db import models
from django.contrib.auth.models import User
from userprofile import UserProfile
from wcs import *
from datetime import datetime

class DiskFile(models.Model):
    file_hash = models.CharField(max_length=40, unique=True, primary_key=True)
    size = models.PositiveIntegerField()
    file_type = models.CharField(max_length=16, null=True)

    @staticmethod
    def get_file_directory(file_hash_digest):
        file_path = '/'.join((file_hash_digest[0:2],
                              file_hash_digest[2:4],
                              file_hash_digest[4:6])
        )
        file_path = DATADIR + file_path + '/'
        return file_path

    def __init__(self, fn):
        pass

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

class Job(models.Model):
    calibration = models.ForeignKey('Calibration')
    
    STATUS_CHOICES = (('S', 'Success'), 
              ('F', 'Failure'))    
    
    status = models.CharField(max_length=1, choices=STATUS_CHOICES)
    error_message = models.CharField(max_length=256)

class UserImage(models.Model):
    image = models.ForeignKey(Image)
    
    PERMISSION_CHOICES = (('pu', 'Public'),
             ('pr', 'Private'))

    permission = models.CharField(max_length=2, choices=PERMISSION_CHOICES)
    tags = models.ManyToManyField('Tag', related_name='images')
    description = models.CharField(max_length=1024)
    original_file_name = models.CharField(max_length=256)
    submission = models.ForeignKey('Submission', related_name='user_images')

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

    DATASRC_CHOICES = (
        ('url', 'URL'),
        ('file', 'File'),
    )

    FILETYPE_CHOICES = (
        ('image', 'Image (jpeg, png, gif, tiff, raw, or FITS)'),
        ('fits', 'FITS table of source locations'),
        ('text', 'Text list of source locations'),
    )

    ###
    disk_file = models.ForeignKey(DiskFile, related_name='submissions')
    data_src = models.CharField(max_length=10, choices=DATASRC_CHOICES)
    # image / fits / text
    file_type = models.CharField(max_length=10, choices=FILETYPE_CHOICES)
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
