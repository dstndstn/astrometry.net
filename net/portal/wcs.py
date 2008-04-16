from django.db import models

from astrometry.util import sip
import math

class TanWCS(models.Model):
    crval1 = models.FloatField()
    crval2 = models.FloatField()
    crpix1 = models.FloatField()
    crpix2 = models.FloatField()
    cd11 = models.FloatField()
    cd12 = models.FloatField()
    cd21 = models.FloatField()
    cd22 = models.FloatField()
    imagew = models.FloatField()
    imageh = models.FloatField()

    def __init__(self, *args, **kwargs):
        filename = None
        if 'file' in kwargs:
            filename = kwargs['file']
            del kwargs['file']
        super(TanWCS, self).__init__(*args, **kwargs)
        if filename:
            wcs = sip.Tan(filename)
            self.crval1 = wcs.crval[0]
            self.crval2 = wcs.crval[1]
            self.crpix1 = wcs.crpix[0]
            self.crpix2 = wcs.crpix[1]
            self.cd11 = wcs.cd[0]
            self.cd12 = wcs.cd[1]
            self.cd21 = wcs.cd[2]
            self.cd22 = wcs.cd[3]
            self.imagew = wcs.imagew
            self.imageh = wcs.imageh

    def __str__(self):
        return ('<TanWCS: CRVAL (%f, %f)' % (self.crval1, self.crval2) +
                ' CRPIX (%f, %f)' % (self.crpix1, self.crpix2) +
                ' CD (%f, %f; %f %f)' % (self.cd11, self.cd12, self.cd21, self.cd22) +
                ' Image size (%f, %f)>' % (self.imagew, self.imageh)
                )

    # returns pixel scale in arcseconds per pixel
    def get_pixscale(self):
        return 3600.0 * math.sqrt(abs(self.cd11 * self.cd22 - self.cd12 * self.cd21))

    # returns the field area in square degrees.
    def get_field_area(self):
        scale = self.get_pixscale() / 3600.0
        return self.imagew * self.imageh * (scale**2)

    def get_field_radius(self):
        area = self.get_field_area()
        return math.sqrt(area) / 2.;

    # returns (ra,dec) in degrees
    def get_field_center(self):
        tan = self.to_tanwcs()
        return tan.pixelxy2radec(self.imagew/2., self.imageh/2.)

    # returns (w, h, units)
    def get_field_size(self):
        scale = self.get_pixscale()
        (fieldw, fieldh) = (self.imagew * scale, self.imageh * scale)
        units = 'arcsec'
        if min(fieldw, fieldh) > 3600:
            fieldw /= 3600.
            fieldh /= 3600.
            units = 'deg'
        elif min(fieldw, fieldh) > 60:
            fieldw /= 60.
            fieldh /= 60.
            units = 'arcmin'
        return (fieldw, fieldh, units)
    
    def radec_bounds(self, nsteps=10):
        tanwcs = self.to_tanwcs()
        return tanwcs.radec_bounds(nsteps)

    def to_tanwcs(self):
        tan = sip.Tan()
        tan.crval[0] = self.crval1
        tan.crval[1] = self.crval2
        tan.crpix[0] = self.crpix1
        tan.crpix[1] = self.crpix2
        tan.cd[0] = self.cd11
        tan.cd[1] = self.cd12
        tan.cd[2] = self.cd21
        tan.cd[3] = self.cd22
        tan.imagew = self.imagew
        tan.imageh = self.imageh
        return tan

class  SipWCS(TanWCS):
    order = models.PositiveSmallIntegerField(default=2)
    terms = models.TextField(default='')
    #def to_python(self, value):
    #vals = []
    #for i in range(self.order+1):
    #    for j in range(self.order+1):
    #        if i + j <= 1:
    #            continue
    #return map(float, vals.split(','))

