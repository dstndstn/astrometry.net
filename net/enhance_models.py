import numpy as np

import fitsio

from django.db import models

import settings #from astrometry.net.settings import *
from astrometry.net.models import *
from astrometry.net.wcs import *

from astrometry.util.util import *
from astrometry.util.starutil_numpy import *
from astrometry.util.resample import *
from astrometry.util.miscutils import *

class EnhanceVersion(models.Model):
    name = models.CharField(max_length=64)
    topscale = models.FloatField()

class EnhancedImage(models.Model):
    version = models.ForeignKey('EnhanceVersion')

    nside = models.IntegerField()
    healpix = models.IntegerField()

    wcs = models.ForeignKey('TanWCS')

    cals = models.ManyToManyField('Calibration',
                                  related_name='enhanced_images',
                                  db_table='enhancedimage_calibration')

    #image  = models.ForeignKey('DiskFile')
    #weight = models.ForeignKey('DiskFile')

    def __str__(self):
        return 'EnhancedImage(ver %s, nside %i, hp %i)' % (self.version.name,
                                                           self.nside, self.healpix)

    def get_dir(self):
        return os.path.join(settings.ENHANCE_DIR, self.version.name,
                            'nside-%i' % self.nside, 'hpk-%i' % (self.healpix / 1000),
                            'hp-%i' % (self.healpix))

    def get_image_path(self):
        return os.path.join(self.get_dir(), 'enhance-image.fits')
    def get_weight_path(self):
        return os.path.join(self.get_dir(), 'enhance-weight.fits')
        
    def init(self):
        hp = self.healpix
        nside = self.nside

        topscale = self.version.topscale

        # healpix corners
        xyz0 = np.array(healpix_to_xyz(hp, nside, 0., 0.))
        xyz1 = np.array(healpix_to_xyz(hp, nside, 1., 0.))
        xyz2 = np.array(healpix_to_xyz(hp, nside, 0., 1.))
        xyz3 = np.array(healpix_to_xyz(hp, nside, 1., 1.))

        # healpix center
        r0,d0 = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
        # unit vector tangent to center, in direction of +Dec
        up = np.array([-np.sin(np.deg2rad(d0))*np.cos(deg2rad(r0)),
                        -np.sin(np.deg2rad(d0))*np.sin(deg2rad(r0)),
                        np.cos(np.deg2rad(d0))])
        # find the angle between +Dec and the healpix edge (0,0)--(1,0)
        d1 = xyz1 - xyz0
        d1 /= np.sqrt(np.sum(d1**2))
        theta = np.pi/2. - np.arccos(np.sum(d1 * up))

        pscale = topscale / nside
        pscale /= 3600.
        wcs = Tan(r, d, 500., 500.,
                  pscale * np.cos(theta), -pscale * np.sin(theta),
                  pscale * np.sin(theta),  pscale * np.cos(theta),
                  1000,1000)
            
        xy = np.array([wcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                       [xyz0, xyz1, xyz2, xyz3]])
        # ok,x,y -> x,y
        xy = xy[:,1:]
        xlo,ylo = xy.min(axis=0)
        xhi,yhi = xy.max(axis=0)
        
        # Make the WCS just barely contain the healpix -- magic 501
        # here comes from 500 above.
        W,H = int(np.ceil(xhi - xlo)), int(np.ceil(yhi - ylo))

        wcsargs = (r, d, 501.-xlo, 501.-ylo,
                   pscale * np.cos(theta), -pscale * np.sin(theta),
                   pscale * np.sin(theta),  pscale * np.cos(theta),
                   float(W), float(H))
        hpwcs = Tan(*wcsargs)

        hpxy = np.array([hpwcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                         [xyz0, xyz1, xyz3, xyz2]])
        hpxy = hpxy[:,1:]
    
        bands = 3

        enhI = np.zeros((H,W,bands), np.float32)
        enhW = np.zeros((H,W), np.float32)
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        # inside-healpix mask.
        enhM = point_in_poly(xx, yy, hpxy-1.)
        del xx
        del yy
        npix = np.sum(enhM)
        for b in range(bands):
            enhI[:,:,b][enhM] = np.random.permutation(npix)
        enhW[enhM] = 1e-3

        mydir = self.get_dir()
        if not os.path.exists(mydir):
            try:
                os.makedirs(mydirs)
            except:
                pass

        imfn = self.get_image_path()
        fitsio.write(imfn, enhI)
        print 'Wrote', imfn

        wfn = self.get_weight_path()
        fitsio.write(wfn, enhW)
        print 'Wrote', wfn

        #**dict(zip(['crval1','crval2','crpix1','crpix2',
        #'cd11','cd12','cd21','cd22','imagew','imageh'], wcsargs)))

        dbwcs = TanWCS()
        dbwcs.set_from_tanwcs(hpwcs)
        print 'Database WCS:', dbwcs
        dbwcs.save()

        self.wcs = dbwcs
