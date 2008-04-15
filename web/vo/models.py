from django.db import models
from django.contrib.auth.models import User

from an.util import orderby
from an.portal.wcs import TanWCS
from an.portal.job import Job

import logging

class Image(models.Model):
    objects = orderby.OrderByManager()

    job = models.ForeignKey(Job, null=True)

    # HACK
    user = models.ForeignKey(User)

    # a short (usually one line) description of the image
    # identifying the image source (e.g., survey name), object name or field coordinates,
    # bandpass/filter, and so forth.
    image_title = models.CharField(max_length=256)

    # the MIME-type of the object associated with the image acref, e.g., "image/fits", "text/html", and so forth.
    #image_format = models.CharField(maxlength=32)

    # specifying the URL to be used to access or retrieve the image.
    #image_url = models.URLField(null=True)

    # the actual or estimated size of the encoded image in bytes (not pixels!). This is useful for image selection
    # and for optimizing distributed computations.
    #image_filesize = models.PositiveIntegerField()

    # the instrument or instruments used to make the observation, e.g., STScI.HST.WFPC2.
    instrument = models.CharField(max_length=256, null=True)

    # the mean modified Julian date of the observation.
    # By "mean" we mean the midpoint of the observation in terms of normalized exposure times:
    # this is the "characteristic observation time" and is independent of observation duration.
    #jdate = models.PositiveIntegerField()

    # WCS
    #wcs = models.ForeignKey(TanWCS)

    # Cached WCS values:

    # (RA,Dec) of image center
    ra_center = models.FloatField()
    dec_center = models.FloatField()

    ra_min = models.FloatField()
    ra_max = models.FloatField()
    dec_min = models.FloatField()
    dec_max = models.FloatField()

    def __str__(self):
        return ('<vo.Image: ' + str(self.field) +
                ' title: "%s"' % self.image_title +
                ' instrument: "%s"' % self.instrument +
                ' jdate: "%i"' % self.jdate +
                ' WCS: ' + str(self.wcs) +
                ' Center (%f, %f)' % (self.ra_center, self.dec_center) +
                ' Range ([%f, %f], [%f, %f])' %
                (self.ra_min, self.ra_max, self.dec_min, self.dec_max) +
                '>')
