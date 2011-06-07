from django.db import models
from django.contrib.auth.models import User
from userprofile import UserProfile
from wcs import *

class DiskFile(models.Model):
    file_hash = models.CharField(max_length=40, unique=True, primary_key=True)
    size = models.PositiveIntegerField()
    file_type = models.CharField(max_length=16, null=True)

class Image(models.Model):
    disk_file = models.ForeignKey(DiskFile)
    width = models.PositiveIntegerField(null=True)
    height = models.PositiveIntegerField(null=True)

class Tag(models.Model):
    user = models.ForeignKey(User)
    text = models.CharField(max_length=4096)
    added_time = models.DateTimeField(auto_now=True) 

class Job(models.Model):
    calibration = models.ForeignKey(Calibration)
    
    STATUS_CHOICES = (('S', 'Success'), 
              ('F', 'Failure'))    
    
    status = models.CharField(max_length=1, choices=STATUS_CHOICES)
    error_message = models.CharField(max_length=256)

 
class Calibration(models.Model):
    # TAN WCS, straight from the quad match
    raw_tan = models.ForeignKey(TanWCS, related_name='calibrations_raw', null=True)
    # TAN WCS, after tweaking
    tweaked_tan = models.ForeignKey(TanWCS, related_name='calibrations_tweaked', null=True)
    # SIP
    sip = models.ForeignKey(SipWCS, null=True)

    # RA,Dec bounding box.
    ramin  = models.FloatField()
    ramax  = models.FloatField()
    decmin = models.FloatField()
    decmax = models.FloatField()

class UserImage(models.Model):
    image = models.ForeignKey(Image)
    
    PERMISSION_CHOICES = (('pu', 'Public'),
             ('pr', 'Private'))

    permission = models.CharField(max_length=2, choices=PERMISSION_CHOICES)
    tags = models.ManyToManyField(Tag, related_name='images')
    description = models.CharField(max_length=1024)
    original_file_name = models.CharField(max_length=256)
    submission = models.ForeignKey(Submission, related_name='user_images')

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
    
	datasrc = models.CharField(max_length=10, choices=DATASRC_CHOICES)

	# image / fits / text
	filetype = models.CharField(max_length=10, choices=FILETYPE_CHOICES)

	url = models.URLField(blank=True, null=True)

	parity = models.PositiveSmallIntegerField(choices=PARITY_CHOICES, default=2)
    
    scale_units = models.CharField(max_length=20, choices=SCALEUNITS_CHOICES)
    scale_type = models.CharField(max_length=2, choices=SCALETYPE_CHOICES)
