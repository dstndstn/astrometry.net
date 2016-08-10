import os
import numpy as np
import tempfile

import fitsio

from django.db import models
from django.db import transaction

import settings

from astrometry.util.util import *

import logging

log = logging.getLogger('enhance_models')

class EnhanceVersion(models.Model):
    name = models.CharField(max_length=64)
    topscale = models.FloatField()

    def __str__(self):
        return 'EnhanceVersion(%s)' % self.name

    def __repr__(self):
        return ('EnhanceVersion(name="%s", topscale=%g, id=%i)' %
                (self.name, self.topscale, self.id))

class EnhancedImage(models.Model):
    version = models.ForeignKey('EnhanceVersion')

    nside = models.IntegerField()
    healpix = models.IntegerField()

    wcs = models.ForeignKey('TanWCS', null=True, default=None)

    cals = models.ManyToManyField('Calibration',
                                  related_name='enhanced_images',
                                  db_table='enhancedimage_calibration')

    maxweight = models.FloatField(default=0.)

    def __str__(self):
        return 'EnhancedImage(ver %s, nside %i, hp %i)' % (self.version.name,
                                                           self.nside, self.healpix)

    def get_dir(self):
        return os.path.join(settings.ENHANCE_DIR, self.version.name,
                            'nside-%i' % self.nside, 'hpk-%i' % (self.healpix / 1000))
    #'hp-%i' % (self.healpix))

    # def get_image_path(self):
    #     return os.path.join(self.get_dir(), 'enhance-image.fits')
    # def get_weight_path(self):
    #     return os.path.join(self.get_dir(), 'enhance-weight.fits')

    def get_imw_path(self):
        return os.path.join(self.get_dir(), 'enhance-hp%i.fits' % self.healpix)

    def read_files(self):
        # imfn = self.get_image_path()
        # wfn = self.get_weight_path()
        # log.debug('Reading files %s and %s' % (imfn, wfn))
        # enhI = fitsio.read(self.get_image_path())
        # enhW = fitsio.read(self.get_weight_path())
        # return enhI, enhW
        fn = self.get_imw_path()
        log.debug('Reading %s' % fn)
        fits = fitsio.FITS(fn, 'r')
        print('Read', len(fits), 'HDUs')
        enhI = fits[0].read()
        enhW = fits[1].read()
        log.debug('Read image %s and weight %s' % (str(enhI.shape), str(enhW.shape)))
        return enhI, enhW

    def write_files(self, enhI, enhW, temp=False):
        if temp:
            mydir = self.get_dir()
            f,fn = tempfile.mkstemp(dir=mydir, suffix='.tmp')
            os.close(f)
        else:
            fn = self.get_imw_path()

        fits = fitsio.FITS(fn, 'rw', clobber=True)
        fits.write_image(enhI)
        fits.write_image(enhW)
        fits.close()
        return fn
        #imfn = self.get_image_path()
        #fitsio.write(imfn, enhI, clobber=True, compress='GZIP')
        #wfn = self.get_weight_path()
        #fitsio.write(wfn, enhW, clobber=True, compress='GZIP')

    def move_temp_files(self, tempfn):
        #imfn = self.get_image_path()
        #wfn = self.get_weight_path()
        #os.rename(imfn + '.tmp', imfn)
        #os.rename(wfn + '.tmp', wfn)
        fn = self.get_imw_path()
        os.rename(tempfn, fn)

    def init(self):
        import os
        from astrometry.util.miscutils import point_in_poly
        from astrometry.net.wcs import TanWCS
        from scipy.ndimage.morphology import binary_dilation

        hp = self.healpix
        nside = self.nside
        topscale = self.version.topscale

        hpwcs,hpxy = get_healpix_wcs(nside, hp, topscale)
        H,W = int(hpwcs.get_height()), int(hpwcs.get_width())

        bands = 3

        enhI = np.zeros((H,W,bands), np.float32)
        enhW = np.zeros((H,W), np.float32)
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        # inside-healpix mask.
        enhM = point_in_poly(xx, yy, hpxy-1.)
        # grow ?
        enhM = binary_dilation(enhM, np.ones((3,3)))

        del xx
        del yy
        npix = np.sum(enhM)
        for b in range(bands):
            enhI[:,:,b][enhM] = np.random.permutation(npix) / float(npix)
        enhW[enhM] = 1e-3

        mydir = self.get_dir()
        # print 'My directory:', mydir
        if not os.path.exists(mydir):
            # print 'Does not exist'
            try:
                os.makedirs(mydir)
                # print 'Created'
            except:
                import traceback
                print('Failed to create dir:')
                traceback.print_exc()
                pass

        tempfn = self.write_files(enhI, enhW, temp=True)
        dbwcs = TanWCS()
        dbwcs.set_from_tanwcs(hpwcs)
        dbwcs.save()
        with transaction.commit_on_success():
            self.move_temp_files(tempfn)
            self.maxweight = 0.
            self.wcs = dbwcs
            self.save()



def get_healpixes_touching_wcs(tan, nside=None, topscale=256.):
    '''
    tan: TanWCS database object.
    '''
    from astrometry.util.starutil_numpy import degrees_between

    if nside is None:
        # print 'Pixscale', tan.get_pixscale()
        nside = 2 ** int(np.round(np.log2(topscale / tan.get_pixscale())))
        # print 'Nside', nside
        nside = int(np.clip(nside, 1, 2**10))
        # print 'Nside', nside

    r1,d1 = healpix_to_radecdeg(0, nside, 0., 0.)
    r2,d2 = healpix_to_radecdeg(0, nside, 0.5, 0.5)
    hpradius = degrees_between(r1,d1, r2,d2)
    # HACK -- padding for squished parallelograms
    hpradius *= 1.5

    r,d,radius = tan.get_center_radecradius()
    radius = np.hypot(radius, hpradius)
    hh = healpix_rangesearch_radec(r, d, radius, nside)
    hh.sort()
    #print 'Healpixes:', hh
    return (nside, hh)

        
def get_healpix_wcs(nside, hp, topscale):
    '''
    Returns a WCS object for the given healpix
    '''
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
    wcs = Tan(r0, d0, 500., 500.,
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

    hpwcs = Tan(r0, d0, 501.-xlo, 501.-ylo,
                pscale * np.cos(theta), -pscale * np.sin(theta),
                pscale * np.sin(theta),  pscale * np.cos(theta),
                float(W), float(H))

    hpxy = np.array([hpwcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                     [xyz0, xyz1, xyz3, xyz2]])
    hpxy = hpxy[:,1:]

    return hpwcs, hpxy

