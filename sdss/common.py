# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import os
from astrometry.util.fits import fits_table
from astrometry.util.miscutils import get_overlapping_region
import numpy as np
from functools import reduce

try:
    import cutils
except:
    cutils = None

cas_flags = dict(
    CANONICAL_CENTER = 0x0000000000000001,
    BRIGHT = 0x0000000000000002,
    EDGE = 0x0000000000000004,
    BLENDED = 0x0000000000000008,
    CHILD = 0x0000000000000010,
    PEAKCENTER = 0x0000000000000020,
    NODEBLEND = 0x0000000000000040,
    NOPROFILE = 0x0000000000000080,
    NOPETRO = 0x0000000000000100,
    MANYPETRO = 0x0000000000000200,
    NOPETRO_BIG = 0x0000000000000400,
    DEBLEND_TOO_MANY_PEAKS = 0x0000000000000800,
    COSMIC_RAY = 0x0000000000001000,
    MANYR50 = 0x0000000000002000,
    MANYR90 = 0x0000000000004000,
    BAD_RADIAL = 0x0000000000008000,
    INCOMPLETE_PROFILE = 0x0000000000010000,
    INTERP = 0x0000000000020000,
    SATURATED = 0x0000000000040000,
    NOTCHECKED = 0x0000000000080000,
    SUBTRACTED = 0x0000000000100000,
    NOSTOKES = 0x0000000000200000,
    BADSKY = 0x0000000000400000,
    PETROFAINT = 0x0000000000800000,
    TOO_LARGE = 0x0000000001000000,
    DEBLENDED_AS_PSF = 0x0000000002000000,
    DEBLEND_PRUNED = 0x0000000004000000,
    ELLIPFAINT = 0x0000000008000000,
    BINNED1 = 0x0000000010000000,
    BINNED2 = 0x0000000020000000,
    BINNED4 = 0x0000000040000000,
    MOVED = 0x0000000080000000,
    DEBLENDED_AS_MOVING = 0x0000000100000000,
    NODEBLEND_MOVING = 0x0000000200000000,
    TOO_FEW_DETECTIONS = 0x0000000400000000,
    BAD_MOVING_FIT = 0x0000000800000000,
    STATIONARY = 0x0000001000000000,
    PEAKS_TOO_CLOSE = 0x0000002000000000,
    MEDIAN_CENTER = 0x0000004000000000,
    LOCAL_EDGE = 0x0000008000000000,
    BAD_COUNTS_ERROR = 0x0000010000000000,
    BAD_MOVING_FIT_CHILD = 0x0000020000000000,
    DEBLEND_UNASSIGNED_FLUX = 0x0000040000000000,
    SATUR_CENTER = 0x0000080000000000,
    INTERP_CENTER = 0x0000100000000000,
    DEBLENDED_AT_EDGE = 0x0000200000000000,
    DEBLEND_NOPEAK = 0x0000400000000000,
    PSF_FLUX_INTERP = 0x0000800000000000,
    TOO_FEW_GOOD_DETECTIONS = 0x0001000000000000,
    CENTER_OFF_AIMAGE = 0x0002000000000000,
    DEBLEND_DEGENERATE = 0x0004000000000000,
    BRIGHTEST_GALAXY_CHILD = 0x0008000000000000,
    CANONICAL_BAND = 0x0010000000000000,
    AMOMENT_FAINT = 0x0020000000000000,
    AMOMENT_SHIFT = 0x0040000000000000,
    AMOMENT_MAXITER = 0x0080000000000000,
    MAYBE_CR = 0x0100000000000000,
    MAYBE_EGHOST = 0x0200000000000000,
    NOTCHECKED_CENTER = 0x0400000000000000,
    OBJECT2_HAS_SATUR_DN = 0x0800000000000000,
    OBJECT2_DEBLEND_PEEPHOLE = 0x1000000000000000,
    GROWN_MERGED = 0x2000000000000000,
    HAS_CENTER = 0x4000000000000000,
    RESERVED = 0x8000000000000000,
    )

# From:
# http://www.sdss3.org/svn/repo/idlutils/trunk/data/sdss/sdssMaskbits.par
# via
#     s = open('sdssMaskbits.par').read()
#     bits = []
#     for line in s.split('\n'):
#       sp = line.split()
#       line = (int(sp[2]), sp[3], ' '.join(sp[4:]))
#       bits.append(line)
#       print repr(bits).replace('), ', '),\n ')
#
photo_flags1_info = [
    # masktype OBJECT1 32 "Object flags from photo reductions for SDSS (first 32)"
    (0, 'CANONICAL_CENTER', '"The quantities (psf counts, model fits and likelihoods) that are usually determined at an object\'s center as determined band-by-band were in fact determined at the canonical center (suitably transformed). This is due to the object being to close to the edge to extract a profile at the local center, and OBJECT1_EDGE is also set."'),
    (1, 'BRIGHT', '"Indicates that the object was detected as a bright object. Since these are typically remeasured as faint objects, most users can ignore BRIGHT objects."'),
    (2, 'EDGE', '"Object is too close to edge of frame in this band."'),
    (3, 'BLENDED', '"Object was determined to be a blend. The flag is set if: more than one peak is detected within an object in a single band together; distinct peaks are found when merging different colours of one object together; or distinct peaks result when merging different objects together. "'),
    (4, 'CHILD', '"Object is a child, created by the deblender."'),
    (5, 'PEAKCENTER', '"Given center is position of peak pixel, as attempts to determine a better centroid failed."'),
    (6, 'NODEBLEND', '"Although this object was marked as a blend, no deblending was attempted."'),
    (7, 'NOPROFILE', '"Frames couldn\'t extract a radial profile."'),
    (8, 'NOPETRO', '" No Petrosian radius or other Petrosian quanties could be measured."'),
    (9, 'MANYPETRO', '"Object has more than one possible Petrosian radius."'),
    (10, 'NOPETRO_BIG', '"The Petrosian ratio has not fallen to the value at which the Petrosian radius is defined at the outermost point of the extracted radial profile. NOPETRO is set, and the Petrosian radius is set to the outermost point in the profile."'),
    (11, 'DEBLEND_TOO_MANY_PEAKS', '"The object had the OBJECT1_DEBLEND flag set, but it contained too many candidate children to be fully deblended. This flag is only set in the parent, i.e. the object with too many peaks."'),
    (12, 'CR', '"Object contains at least one pixel which was contaminated by a cosmic ray. The OBJECT1_INTERP flag is also set. This flag does not mean that this object is a cosmic ray; rather it means that a cosmic ray has been removed. "'),
    (13, 'MANYR50', '" More than one radius was found to contain 50% of the Petrosian flux. (For this to happen part of the radial profile must be negative)."'),
    (14, 'MANYR90', '"More than one radius was found to contain 90% of the Petrosian flux. (For this to happen part of the radial profile must be negative)."'),
    (15, 'BAD_RADIAL', '" Measured profile includes points with a S/N <= 0. In practice this flag is essentially meaningless."'),
    (16, 'INCOMPLETE_PROFILE', '"A circle, centerd on the object, of radius the canonical Petrosian radius extends beyond the edge of the frame. The radial profile is still measured from those parts of the object that do lie on the frame."'),
    (17, 'INTERP', '" The object contains interpolated pixels (e.g. cosmic rays or bad columns)."'),
    (18, 'SATUR', '"The object contains saturated pixels; INTERP is also set."'),
    (19, 'NOTCHECKED', '"Object includes pixels that were not checked for peaks, for example the unsmoothed edges of frames, and the cores of subtracted or saturated stars."'),
    (20, 'SUBTRACTED', '"Object (presumably a star) had wings subtracted."'),
    (21, 'NOSTOKES', '"Object has no measured Stokes parameters."'),
    (22, 'BADSKY', '"The estimated sky level is so bad that the central value of the radial profile is crazily negative; this is usually the result of the subtraction of the wings of bright stars failing."'),
    (23, 'PETROFAINT', '"At least one candidate Petrosian radius occured at an unacceptably low surface brightness."'),
    (24, 'TOO_LARGE', '" The object is (as it says) too large. Either the object is still detectable at the outermost point of the extracted radial profile (a radius of approximately 260 arcsec), or when attempting to deblend an object, at least one child is larger than half a frame (in either row or column)."'),
    (25, 'DEBLENDED_AS_PSF', '"When deblending an object, in this band this child was treated as a PSF."'),
    (26, 'DEBLEND_PRUNED', '"When solving for the weights to be assigned to each child the deblender encountered a nearly singular matrix, and therefore deleted at least one of them."'),
    (27, 'ELLIPFAINT', '"No isophotal fits were performed."'),
    (28, 'BINNED1', '"The object was detected in an unbinned image."'),
    (29, 'BINNED2', '" The object was detected in a 2x2 binned image after all unbinned detections have been replaced by the background level."'),
    (30, 'BINNED4', '"The object was detected in a 4x4 binned image. The objects detected in the 2x2 binned image are not removed before doing this."'),
    (31, 'MOVED', '"The object appears to have moved during the exposure. Such objects are candidates to be deblended as moving objects."'),
]

photo_flags2_info = [
  (0, 'DEBLENDED_AS_MOVING', '"The object has the MOVED flag set, and was deblended on the assumption that it was moving."'),
  (1, 'NODEBLEND_MOVING', '"The object has the MOVED flag set, but was not deblended as a moving object."'),
  (2, 'TOO_FEW_DETECTIONS', '"The object has the MOVED flag set, but has too few detection to be deblended as moving."'),
  (3, 'BAD_MOVING_FIT', '"The fit to the object as a moving object is too bad to be believed."'),
  (4, 'STATIONARY', '"A moving objects velocity is consistent with zero"'),
  (5, 'PEAKS_TOO_CLOSE', '"Peaks in object were too close (set only in parent objects)."'),
  (6, 'BINNED_CENTER', '"When centroiding the object the object\'s size is larger than the (PSF) filter used to smooth the image."'),
  (7, 'LOCAL_EDGE', '"The object\'s center in some band was too close to the edge of the frame to extract a profile."'),
  (8, 'BAD_COUNTS_ERROR', '"An object containing interpolated pixels had too few good pixels to form a reliable estimate of its error"'),
  (9, 'BAD_MOVING_FIT_CHILD', '"A putative moving child\'s velocity fit was too poor, so it was discarded, and the parent was not deblended as moving"'),
  (10, 'DEBLEND_UNASSIGNED_FLUX', '"After deblending, the fraction of flux assigned to none of the children was too large (this flux is then shared out as described elsewhere)."'),
  (11, 'SATUR_CENTER', '"An object\'s center is very close to at least one saturated pixel; the object may well be causing the saturation."'),
  (12, 'INTERP_CENTER', '"An object\'s center is very close to at least one interpolated pixel."'),
  (13, 'DEBLENDED_AT_EDGE', '"An object so close to the edge of the frame that it would not ordinarily be deblended has been deblended anyway. Only set for objects large enough to be EDGE in all fields/strips."'),
  (14, 'DEBLEND_NOPEAK', '"A child had no detected peak in a given band, but we centroided it anyway and set the BINNED1"'),
  (15, 'PSF_FLUX_INTERP', '"The fraction of light actually detected (as opposed to guessed at by the interpolator) was less than some number (currently 80%) of the total."'),
  (16, 'TOO_FEW_GOOD_DETECTIONS', '"A child of this object had too few good detections to be deblended as moving."'),
  (17, 'CENTER_OFF_AIMAGE', '"At least one peak\'s center lay off the atlas image in some band. This can happen when the object\'s being deblended as moving, or if the astrometry is badly confused."'),
  (18, 'DEBLEND_DEGENERATE', '"At least one potential child has been pruned because its template was too similar to some other child\'s template."'),
  (19, 'BRIGHTEST_GALAXY_CHILD', '"This is the brightest child galaxy in a blend."'),
  (20, 'CANONICAL_BAND', '"This band was the canonical band. This is the band used to measure the Petrosian radius used to calculate the Petrosian counts in each band, and to define the model used to calculate model colors; it has no effect upon the coordinate system used for the OBJC center."'),
  (21, 'AMOMENT_UNWEIGHTED', '"`Adaptive\' moments are actually unweighted."'),
  (22, 'AMOMENT_SHIFT', '"Object\'s center moved too far while determining adaptive moments. In this case, the M_e1 and M_e2 give the (row, column) shift, not the object\'s shape."'),
  (23, 'AMOMENT_MAXITER', '"Too many iterations while determining adaptive moments."'),
  (24, 'MAYBE_CR', '"This object may be a cosmic ray. This bit can get set in the cores of bright stars, and is quite likely to be set for the cores of saturated stars."'),
  (25, 'MAYBE_EGHOST', '"Object appears in the right place to be an electronics ghost."'),
  (26, 'NOTCHECKED_CENTER', '"Center of object lies in a NOTCHECKED region. The object is almost certainly bogus."'),
  (27, 'HAS_SATUR_DN', '"This object is saturated in this band and the bleed trail doesn\'t touch the edge of the frame, we we\'ve made an attempt to add up all the flux in the bleed trails, and to include it in the object\'s photometry. "'),
  (28, 'DEBLEND_PEEPHOLE', '"The deblend was modified by the optimizer"'),
  (29, 'SPARE3', '""'),
  (30, 'SPARE2', '""'),
  (31, 'SPARE1', '""'),
]


specobj_boss_target1_info = [
    # masktype BOSS_TARGET1 64 "BOSS survey primary target selection flags"
    # galaxies
    (0, 'GAL_LOZ', "low-z lrgs"),
    (1, 'GAL_CMASS', "dperp > 0.55, color-mag cut"),
    (2, 'GAL_CMASS_COMM', "dperp > 0.55, commissioning color-mag cut"),
    (3, 'GAL_CMASS_SPARSE', "GAL_CMASS_COMM & (!GAL_CMASS) & (i < 19.9) sparsely sampled"),
    (6, 'SDSS_KNOWN', "Matches a known SDSS spectra"),
    (7, 'GAL_CMASS_ALL', "GAL_CMASS and the entire sparsely sampled region"),
    (8, 'GAL_IFIBER2_FAINT', "ifiber2 > 21.5, extinction corrected. Used after Nov 2010"),
    # galaxies deprecated
    #maskbits BOSS_TARGET1 3 GAL_GRRED "red in g-r"
    #maskbits BOSS_TARGET1 4 GAL_TRIANGLE "GAL_HIZ and !GAL_CMASS"
    #maskbits BOSS_TARGET1 5 GAL_LODPERP "Same as hiz but between dperp00 and dperp0"
    # qsos (1)
    (10, 'QSO_CORE', "restrictive qso selection: commissioning only"),
    (11, 'QSO_BONUS', "permissive qso selection: commissioning only"),
    (12, 'QSO_KNOWN_MIDZ', "known qso between [2.2,9.99]"),
    (13, 'QSO_KNOWN_LOHIZ', "known qso outside of miz range. never target"),
    (14, 'QSO_NN', "Neural Net that match to sweeps/pass cuts"),
    (15, 'QSO_UKIDSS', "UKIDSS stars that match sweeps/pass flag cuts"),
    (16, 'QSO_KDE_COADD', "kde targets from the stripe82 coadd"),
    (17, 'QSO_LIKE', "likelihood method"),
    (18, 'QSO_FIRST_BOSS', "FIRST radio match"),
    (19, 'QSO_KDE', "selected by kde+chi2"),
    # standards
    (20, 'STD_FSTAR', "standard f-stars"),
    (21, 'STD_WD', "white dwarfs"),
    (22, 'STD_QSO', "qso"),
    # template stars
    (32, 'TEMPLATE_GAL_PHOTO', "galaxy templates"),
    (33, 'TEMPLATE_QSO_SDSS1', "QSO templates"),
    (34, 'TEMPLATE_STAR_PHOTO', "stellar templates"),
    (35, 'TEMPLATE_STAR_SPECTRO', "stellar templates (spectroscopically known)"),
    # qsos (2)
    (40, 'QSO_CORE_MAIN', "Main survey core sample"),
    (41, 'QSO_BONUS_MAIN', "Main survey bonus sample"),
    (42, 'QSO_CORE_ED', "Extreme Deconvolution in Core"),
    (43, 'QSO_CORE_LIKE', "Likelihood that make it into core"),
    (44, 'QSO_KNOWN_SUPPZ', "known qso between [1.8,2.15]"),
]


specobj_boss_target1_map = dict([(nm, 1<<bit)
                                 for bit,nm,desc in specobj_boss_target1_info])

photo_flags1_map = dict([(nm, 1<<bit)
                         for bit,nm,desc in photo_flags1_info])
photo_flags2_map = dict([(nm, 1<<bit)
                         for bit,nm,desc in photo_flags2_info])


def band_names():
    return ['u','g','r','i','z']

def band_name(b):
    if b in band_names():
        return b
    if b in [0,1,2,3,4]:
        return 'ugriz'[b]
    raise Exception('Invalid SDSS band: "' + str(b) + '"')

def band_index(b):
    if b in band_names():
        return 'ugriz'.index(b)
    if b in [0,1,2,3,4]:
        return b
    raise Exception('Invalid SDSS band: "' + str(b) + '"')

class SdssDR(object):
    def __init__(self, curl=False, basedir=None):
        self.curl = curl
        self.basedir = basedir
        self.filenames = {}

    def getDRNumber(self):
        return -1
        
    def getFilename(self, filetype, *args, **kwargs):
        for k,v in zip(['run', 'camcol', 'field', 'band'], args):
            kwargs[k] = v
        # convert band number to band character.
        if 'band' in kwargs and kwargs['band'] is not None:
            kwargs['band'] = band_name(kwargs['band'])
        if not filetype in self.filenames:
            return None
        pat = self.filenames[filetype]
        if kwargs.get('rerun', None) is None:
            run = kwargs.get('run', None)
            rerun = self.get_rerun(run)
            kwargs.update(rerun=rerun)
        fn = pat % kwargs
        return fn

    def get_rerun(self, run, field=None):
        return None

    def getPath(self, *args, **kwargs):
        fn = self.getFilename(*args, **kwargs)
        if fn is None:
            return None
        if self.basedir is not None:
            fn = os.path.join(self.basedir, fn)
        return fn

    def setBasedir(self, dirnm):
        self.basedir = dirnm

    def _open(self, fn):

        if self.basedir is not None:
            path = os.path.join(self.basedir, fn)
        else:
            path = fn

        try:
            import fitsio
            return fitsio_wrapper(fitsio.FITS(path))
        except ImportError:
            pass

        try:
            import pyfits
        except ImportError:
            try:
                from astropy.io import fits as pyfits
            except ImportError:
                raise ImportError("Cannot import either pyfits or astropy.io.fits")
        return pyfits.open(path)

class fitsio_wrapper(object):
    def __init__(self, F):
        self.F = F
    def __getitem__(self, k):
        hdu = self.F[k]
        hdu.data = hdu
        return hdu
    
class SdssFile(object):
    def __init__(self, run=None, camcol=None, field=None, band=None, rerun=None,
                 **kwargs):
        '''
        band: string ('u', 'g', 'r', 'i', 'z')
        '''
        self.run = run
        self.camcol = camcol
        self.field = field
        if band is not None:
            self.band = band_name(band)
            self.bandi = band_index(band)
        if rerun is not None:
            self.rerun = rerun
        self.filetype = 'unknown'

    def getRun(self):
        return self.__dict__.get('run', 0)
    def getCamcol(self):
        return self.__dict__.get('camcol', 0)
    def getField(self):
        return self.__dict__.get('field', 0)

    def __str__(self):
        s = 'SDSS ' + self.filetype
        s += ' %i-%i-%i' % (self.getRun(), self.getCamcol(), self.getField())
        if hasattr(self, 'band'):
            s += '-%s' % self.band
        return s


def munu_to_radec_rad(mu, nu, node, incl):
    '''
    Converts SDSS survey coords (mu,nu) into RA,Dec.

    This function requires mu, nu, node, incl to be in RADIANS.

    See munu_to_radec_deg for DEGREES.
    '''
    ra = node + np.arctan2(np.sin(mu - node) * np.cos(nu) * np.cos(incl) -
                           np.sin(nu) * np.sin(incl),
                           np.cos(mu - node) * np.cos(nu))
    dec = np.arcsin(np.sin(mu - node) * np.cos(nu) * np.sin(incl) +
                    np.sin(nu) * np.cos(incl))
    return ra,dec

def munu_to_radec_deg(mu, nu, node, incl):
    '''
    Converts SDSS survey coords (mu,nu) into RA,Dec.

    This function requires mu, nu, node, incl to be in DEGREES.

    See munu_to_radec_rad for RADIANS.
    '''
    mu, nu = np.deg2rad(mu), np.deg2rad(nu)
    node, incl = np.deg2rad(node), np.deg2rad(incl)
    ra,dec = munu_to_radec_rad(mu, nu, node, incl)
    ra, dec = np.rad2deg(ra), np.rad2deg(dec)
    ra += (360. * (ra < 0))
    ra -= (360. * (ra > 360))
    return (ra, dec)


# makes an SDSS AsTrans WCS object look like an anwcs /  Tan / Sip
class AsTransWrapper(object):
    def __init__(self, wcs, w, h, x0=0, y0=0):
        self.wcs = wcs
        self.imagew = w
        self.imageh = h
        self.x0 = x0
        self.y0 = y0
    def pixelxy2radec(self, x, y):
        r,d = self.wcs.pixel_to_radec(x+self.x0-1, y+self.y0-1)
        return r, d
    def radec2pixelxy(self, ra, dec):
        x,y = self.wcs.radec_to_pixel(ra, dec)
        return True, x-self.x0+1, y-self.y0+1

class AsTrans(SdssFile):
    '''
    In DR7, asTrans structures can appear in asTrans files (for a
    whole run) or in tsField files (in astrom/ or fastrom/).

    http://www.sdss.org/dr7/dm/flatFiles/asTrans.html

    In DR8, they are in asTrans files, or in the "frames".

    http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/astrom/asTrans.html
    '''
    def __init__(self, *args, **kwargs):
        '''
        node, incl: in radians

        astrans: must be an object with fields:
         {a,b,c,d,e,f}[band]
         {ricut}[band]
         {drow0, drow1, drow2, drow3, dcol0, dcol1, dcol2, dcol3}[band]
         {csrow, cscol, ccrow, cccol}[band]

        cut_to_band: in DR8 frames files, the astrans elements are not arrays;
        in DR7 tsField files they are.

        Note about units in this class:

        mu,nu are in degrees (great circle coords)

        a,d are in degrees (mu0, nu0)
        b,c,e,f are in degrees/pixel (dmu,dnu/drow,dcol)
        drow0,dcol0 are in pixels (distortion coefficients order 0); dpixels
        drow1,dcol1 are unitless dpixels / pixel (distortion coefficients order 1)
        drow2,dcol2 are in 1/pixels (dpixels/pixel**2) (distortion coefficients order 2)
        drow3,dcol3 are in 1/pixels**2 (dpixels/pixel**3) (distortion coefficients order 3)
        csrow,cscol are in pixels/mag (color-dependent shift)
        ccrow,cccol are in pixels (non-color-dependent shift)
        '''
        super(AsTrans, self).__init__(*args, **kwargs)
        self.filetype = 'asTrans'
        self.node = kwargs.get('node', None)
        self.incl = kwargs.get('incl', None)
        astrans = kwargs.get('astrans', None)
        self.trans = {}
        cut = kwargs.get('cut_to_band', True)
        if astrans is not None and hasattr(self, 'bandi'):
            for f in ['a','b','c','d','e','f', 'ricut',
                      'drow0', 'drow1', 'drow2', 'drow3',
                      'dcol0', 'dcol1', 'dcol2', 'dcol3',
                      'csrow', 'cscol', 'ccrow', 'cccol']:
                try:
                    if hasattr(astrans, f):
                        el = getattr(astrans, f)
                        if cut:
                            el = el[self.bandi]
                        self.trans[f] = el
                except:
                    print('failed to get astrans.' + f)
                    import traceback
                    traceback.print_exc()
                    pass

        self._cache_vals()

    @staticmethod
    def read(fn, F=None, primhdr=None, table=None):
        '''
        F: fitsio.FITS object to use an already-open file.
        primhdr: FITS header object for the primary HDU.
        table: astrometry.util.fits table
        '''
        if F is None:
            import fitsio
            F = fitsio.FITS(fn)
        if primhdr is None:
            primhdr = F[0].read_header()
        band   = primhdr['FILTER'].strip()
        run    = primhdr['RUN']
        camcol = primhdr['CAMCOL']
        field  = 0  # 'FRAME' != field
        if table is None:
            tab = fits_table(F[3].read(lower=True))
        else:
            tab = table
        assert(len(tab) == 1)
        tab = tab[0]
        return AsTrans(run, camcol, field, band,
                       node=np.deg2rad(tab.node),
                       incl=np.deg2rad(tab.incl),
                       astrans=tab, cut_to_band=False)
        
    def __str__(self):
        return (SdssFile.__str__(self) +
                ' (node=%g, incl=%g)' % (self.node, self.incl))

    def _cache_vals(self):
        a, b, c, d, e, f = self._get_abcdef()
        determinant = b * f - c * e
        B =  f / determinant
        C = -c / determinant
        E = -e / determinant
        F =  b / determinant
        py,px, qy,qx = self._get_cscc()
        g0, g1, g2, g3 = self._get_drow()
        h0, h1, h2, h3 = self._get_dcol()
        color0 = self._get_ricut()
        self._cached = [self.node, self.incl,
                        a,b,c,d,e,f, B,C,E,F, px,py,qx,qy, g0,g1,g2,g3,
                        h0,h1,h2,h3, color0]

    def _get_abcdef(self):
        return tuple(self.trans[x] for x in 'abcdef')

    def _get_drow(self):
        return tuple(self.trans[x] for x in ['drow0', 'drow1', 'drow2', 'drow3'])

    def _get_dcol(self):
        return tuple(self.trans[x] for x in ['dcol0', 'dcol1', 'dcol2', 'dcol3'])

    def _get_cscc(self):
        return tuple(self.trans[x] for x in ['csrow', 'cscol', 'ccrow', 'cccol'])

    def _get_ricut(self):
        return self.trans['ricut']

    def cd_at_pixel(self, x, y, color=0):
        '''
        (x,y) to numpy array (2,2) -- the CD matrix at pixel x,y:

        [ [ dRA/dx * cos(Dec), dRA/dy * cos(Dec) ],
          [ dDec/dx          , dDec/dy           ] ]

        in FITS these are called:
        [ [ CD11             , CD12              ],
          [ CD21             , CD22              ] ]

          Note: these statements have not been verified by the FDA.
        '''
        ra0,dec0 = self.pixel_to_radec(x, y, color)
        step = 10. # pixels
        rax,decx = self.pixel_to_radec(x+step, y, color)
        ray,decy = self.pixel_to_radec(x, y+step, color)
        cosd = np.cos(np.deg2rad(dec0))
        return np.array([ [ (rax-ra0)/step * cosd, (ray-ra0)/step * cosd ],
                          [ (decx-dec0)/step     , (decy-dec0)/step      ] ])

    def pixel_to_radec(self, x, y, color=0):
        mu, nu = self.pixel_to_munu(x, y, color)
        return self.munu_to_radec(mu, nu)

    def radec_to_pixel_single_py(self, ra, dec, color=0):
        '''RA,Dec -> x,y for scalar RA,Dec.'''
        # RA,Dec -> mu,nu -> prime -> pixel
        mu, nu = self.radec_to_munu_single(ra, dec)
        return self.munu_to_pixel_single(mu, nu, color)

    def radec_to_pixel_single_c(self, ra, dec):
        return cutils.radec_to_pixel(float(ra), float(dec), self._cached)

    def radec_to_pixel(self, ra, dec, color=0):
        mu, nu = self.radec_to_munu(ra, dec)
        return self.munu_to_pixel(mu, nu, color)
    
    def munu_to_pixel(self, mu, nu, color=0):
        xprime, yprime = self.munu_to_prime(mu, nu, color)
        return self.prime_to_pixel(xprime, yprime, color=color)

    munu_to_pixel_single = munu_to_pixel

    def munu_to_prime(self, mu, nu, color=0):
        '''
        mu = a + b * rowm + c * colm
        nu = d + e * rowm + f * colm

        So

        [rowm; colm] = [b,c; e,f]^-1 * [mu-a; nu-d]

        [b,c; e,f]^1 = [B,C; E,F] in the code below, so

        [rowm; colm] = [B,C; E,F] * [mu-a; nu-d]

        '''
        a, b, c, d, e, f = self._get_abcdef()
        determinant = b * f - c * e
        B =  f / determinant
        C = -c / determinant
        E = -e / determinant
        F =  b / determinant
        mua = mu - a
        # in field 6955, g3, 809 we see a~413
        #if mua < -180.:
        #   mua += 360.
        mua += 360. * (mua < -180.)
        yprime = B * mua + C * (nu - d)
        xprime = E * mua + F * (nu - d)
        return xprime,yprime

    def pixel_to_munu(self, x, y, color=0):
        (xprime, yprime) = self.pixel_to_prime(x, y, color)
        a, b, c, d, e, f = self._get_abcdef()
        mu = a + b * yprime + c * xprime
        nu = d + e * yprime + f * xprime
        return (mu, nu)

    def pixel_to_prime(self, x, y, color=0):
        # Secret decoder ring:
        #  http://www.sdss.org/dr7/products/general/astrometry.html
        # (color)0 is called riCut;
        # g0, g1, g2, and g3 are called
        #    dRow0, dRow1, dRow2, and dRow3, respectively;
        # h0, h1, h2, and h3 are called
        #    dCol0, dCol1, dCol2, and dCol3, respectively;
        # px and py are called csRow and csCol, respectively;
        # and qx and qy are called ccRow and ccCol, respectively.
        color0 = self._get_ricut()
        g0, g1, g2, g3 = self._get_drow()
        h0, h1, h2, h3 = self._get_dcol()
        px, py, qx, qy = self._get_cscc()

        # #$(%*&^(%$%*& bad documentation.
        (px,py) = (py,px)
        (qx,qy) = (qy,qx)

        yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
        xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3

        # The code below implements this, vectorized:
        # if color < color0:
        #   xprime += px * color
        #   yprime += py * color
        # else:
        #   xprime += qx
        #   yprime += qy
        qx = qx * np.ones_like(x)
        qy = qy * np.ones_like(y)
        xprime += np.where(color < color0, px * color, qx)
        yprime += np.where(color < color0, py * color, qy)

        return (xprime, yprime)

    def prime_to_pixel(self, xprime, yprime,  color=0):
        color0 = self._get_ricut()
        g0, g1, g2, g3 = self._get_drow()
        h0, h1, h2, h3 = self._get_dcol()
        px, py, qx, qy = self._get_cscc()

        # #$(%*&^(%$%*& bad documentation.
        (px,py) = (py,px)
        (qx,qy) = (qy,qx)

        qx = qx * np.ones_like(xprime)
        qy = qy * np.ones_like(yprime)
        xprime -= np.where(color < color0, px * color, qx)
        yprime -= np.where(color < color0, py * color, qy)

        # Now invert:
        #   yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
        #   xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
        x = xprime - h0
        # dumb-ass Newton's method
        dx = 1.
        # FIXME -- should just update the ones that aren't zero
        # FIXME -- should put in some failsafe...
        while np.max(np.abs(np.atleast_1d(dx))) > 1e-10:
            xp    = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
            dxpdx = 1 +      h1     + h2 * 2*x +  h3 * 3*x**2
            dx = (xprime - xp) / dxpdx
            x += dx
        y = yprime - (g0 + g1 * x + g2 * x**2 + g3 * x**3)
        return (x, y)

    def radec_to_munu_single_c(self, ra, dec):
        ''' Compute ra,dec to mu,nu for a single RA,Dec, calling C code'''
        mu,nu = cutils.radec_to_munu(ra, dec, self.node, self.incl)
        return mu,nu

    def radec_to_munu(self, ra, dec):
        '''
        RA,Dec in degrees

        mu,nu (great circle coords) in degrees
        '''
        node,incl = self.node, self.incl
        assert(ra is not None)
        assert(dec is not None)
        ra, dec = np.deg2rad(ra), np.deg2rad(dec)
        mu = node + np.arctan2(np.sin(ra - node) * np.cos(dec) * np.cos(incl) +
                               np.sin(dec) * np.sin(incl),
                               np.cos(ra - node) * np.cos(dec))
        nu = np.arcsin(-np.sin(ra - node) * np.cos(dec) * np.sin(incl) +
                       np.sin(dec) * np.cos(incl))
        mu, nu = np.rad2deg(mu), np.rad2deg(nu)
        mu += (360. * (mu < 0))
        mu -= (360. * (mu > 360))
        return (mu, nu)

    def munu_to_radec(self, mu, nu):
        node,incl = self.node, self.incl
        assert(mu is not None)
        assert(nu is not None)
        # just in case you thought we needed *more* rad/deg conversions...
        return munu_to_radec_deg(mu, nu, np.rad2deg(node), np.rad2deg(incl))


if cutils is not None:
    AsTrans.radec_to_munu_single = AsTrans.radec_to_munu_single_c
    AsTrans.radec_to_pixel_single = AsTrans.radec_to_pixel_single_c
else:
    AsTrans.radec_to_munu_single = AsTrans.radec_to_munu
    AsTrans.radec_to_pixel_single = AsTrans.radec_to_pixel_single_py


class TsField(SdssFile):
    def __init__(self, *args, **kwargs):
        super(TsField, self).__init__(*args, **kwargs)
        self.filetype = 'tsField'
        self.exptime = 53.907456
    def setHdus(self, p):
        self.hdus = p
        self.table = fits_table(self.hdus[1].data)[0]
        T = self.table
        self.aa = T.aa.astype(float)
        self.kk = T.kk.astype(float)
        self.airmass = T.airmass

    def getAsTrans(self, band):
        bandi = band_index(band)
        band = band_name(band)
        #node,incl = self.getNode(), self.getIncl()
        hdr = self.hdus[0].header
        node = np.deg2rad(hdr.get('NODE'))
        incl = np.deg2rad(hdr.get('INCL'))
        asTrans = AsTrans(self.run, self.camcol, self.field, band=band,
                          node=node, incl=incl, astrans=self.table)
        return asTrans

    #magL = -(2.5/ln(10))*[asinh((f/f0)/2b)+ln(b)]
    # luptitude == arcsinh mag
    # band: int
    def luptitude_to_counts(self, L, band):
        # from arcsinh softening parameters table
        #   http://www.sdss.org/dr7/algorithms/fluxcal.html#counts2mag
        b = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]

        b = b[band]
        maggies = 2.*b * np.sinh(-0.4 * np.log(10.) * L - np.log(b))
        dlogcounts = -0.4 * (self.aa[band] + self.kk[band] * self.airmass[band])
        return (maggies * self.exptime) * 10.**dlogcounts

    def get_zeropoint(self, band):
        return (2.5 * np.log10(self.exptime)
                -(self.aa[band] + self.kk[band] * self.airmass[band]))
        
    # band: int
    def mag_to_counts(self, mag, band):
        # log_10(counts)
        logcounts = (-0.4 * mag + np.log10(self.exptime)
                     - 0.4*(self.aa[band] + self.kk[band] * self.airmass[band]))
        #logcounts = np.minimum(logcounts, 308.)
        #olderrs = np.seterr(all='print')
        rtn = 10.**logcounts
        #np.seterr(**olderrs)
        return rtn

    def counts_to_mag(self, counts, band):
        # http://www.sdss.org/dr5/algorithms/fluxcal.html#counts2mag
        # f/f0 = counts/exptime * 10**0.4*(aa + kk * airmass)
        # mag = -2.5 * log10(f/f0)
        return -2.5 * (np.log10(counts / self.exptime) +
                       0.4 * (self.aa[band] + self.kk[band] * self.airmass[band]))


    
class FpObjc(SdssFile):
    def __init__(self, *args, **kwargs):
        super(FpObjc, self).__init__(*args, **kwargs)
        self.filetype = 'fpObjc'

class FpM(SdssFile):
    def __init__(self, *args, **kwargs):
        super(FpM, self).__init__(*args, **kwargs)
        self.filetype = 'fpM'
        self.maskmap = None

    def setHdus(self, p):
        self.hdus = p

    def getMaskPlane(self, name):
        # Mask planes are described in HDU 11 (the last HDU)
        if self.maskmap is None:
            self.maskmap = {}
            T = fits_table(self.hdus[-1].data)
            T.cut(T.defname == 'S_MASKTYPE')
            for k,v in zip(T.attributename, T.value):
                k = k.replace('S_MASK_', '')
                if k == 'S_NMASK_TYPES':
                    continue
                self.maskmap[k] = v
        if not name in self.maskmap:
            raise RuntimeError('Unknown mask plane \"%s\"' % name)

        data = self.hdus[1 + self.maskmap[name]].data
        try:
            if data.get_nrows() == 0:
                return None
        except:
            pass
        return fits_table(data)

    def setMaskedPixels(self, name, img, val, roi=None):
        M = self.getMaskPlane(name)
        if M is None:
            return
        if roi is not None:
            x0,x1,y0,y1 = roi

        for (c0,c1,r0,r1,coff,roff) in zip(M.cmin,M.cmax,M.rmin,M.rmax,
                                           M.col0, M.row0):
            assert(coff == 0)
            assert(roff == 0)
            if roi is not None:
                (outx,nil) = get_overlapping_region(c0-x0, c1+1-x0, 0, x1-x0)
                (outy,nil) = get_overlapping_region(r0-y0, r1+1-y0, 0, y1-y0)
                img[outy,outx] = val
            else:
                img[r0:r1+1, c0:c1+1] = val
        

class FpC(SdssFile):
    def __init__(self, *args, **kwargs):
        super(FpC, self).__init__(*args, **kwargs)
        self.filetype = 'fpC'
    def getImage(self):
        return self.image
    def getHeader(self):
        return self.header

class PsField(SdssFile):
    def __init__(self, *args, **kwargs):
        super(PsField, self).__init__(*args, **kwargs)
        self.filetype = 'psField'

    def setHdus(self, p):
        self.hdus = p
        t = fits_table(p[6].data)
        # the table has only one row...
        assert(len(t) == 1)
        t = t[0]
        #self.table = t
        self.gain = t.gain
        self.dark_variance = t.dark_variance
        self.sky = t.sky
        self.skyerr = t.skyerr
        self.psp_status = t.status
        # Double-Gaussian PSF params
        self.dgpsf_s1 = t.psf_sigma1_2g
        self.dgpsf_s2 = t.psf_sigma2_2g
        self.dgpsf_b  = t.psf_b_2g
        # summary PSF width (sigmas)
        self.psf_fwhm = t.psf_width * (2.*np.sqrt(2.*np.log(2.)))

        # 2-gaussian plus power-law PSF params
        self.plpsf_s1 = t.psf_sigma1
        self.plpsf_s2 = t.psf_sigma2
        self.plpsf_b = t.psf_b
        self.plpsf_p0 = t.psf_p0
        self.plpsf_beta = t.psf_beta
        self.plpsf_sigmap = t.psf_sigmap


        t = fits_table(p[8].data)
        self.per_run_apcorrs = t.ap_corr_run

    def getPowerLaw(self, bandnum):
        ''' Returns:

        (a1, sigma_1,
        a2, sigma_2,
        a3, sigma_power, beta_power)

        Where a1 is the amplitude of the first Gaussian and sigma_1 is
        its standard deviation; a2 and sigma_2 are the same for the
        second Gaussian component, and a3 is the amplitude for the
        power-law component.  Sigma is the scale length, beta the
        power.

        RHL claims:
          func = a*[exp(-x^2/(2*sigmax1^2) - y^2/(2*sigmay1^2)) +
                    b*exp(-x^2/(2*sigmax2^2) - y^2/(2*sigmay2^2)) +
                    p0*(1 + r^2/(beta*sigmap^2))^{-beta/2}]

        '''
        return (1., self.plpsf_s1[bandnum],
                self.plpsf_b[bandnum], self.plpsf_s1[bandnum],
                self.plpsf_p0[bandnum], self.plpsf_sigmap[bandnum],
                self.plpsf_beta[bandnum])

    def getPsfFwhm(self, bandnum):
        return self.psf_fwhm[bandnum]

    def getDoubleGaussian(self, bandnum, normalize=False):
        # http://www.sdss.org/dr7/dm/flatFiles/psField.html
        # good = PSP_FIELD_OK
        status = self.psp_status[bandnum]
        if status != 0:
            print('Warning: PsField status[band=%s] =' % (bandnum), status)

        # b is the "ratio of G2 to G1 at the origin", ie, not the
        # straight Gaussian amplitudes
        a  = 1.0
        s1 = self.dgpsf_s1[bandnum]
        s2 = self.dgpsf_s2[bandnum]
        b  = self.dgpsf_b[bandnum]

        # value at center is 1./(2.*pi*sigma**2)

        if normalize:
            b *= (s2/s1)**2
            absum = (a + b)
            a /= absum
            b /= absum
        
        return (float(a), float(s1), float(b), float(s2))
 
    def getEigenPsfs(self, bandnum):
        '''
        Returns a numpy array of shape, eg, (4, 51, 51).
        '''
        T = fits_table(self.hdus[bandnum+1].data)
        psfs = []
        for psf,h,w in zip(T.rrows, T.rnrow, T.rncol):
            psfs.append(psf.reshape((h,w)))
        psfs = np.array(psfs)
        return psfs

    def getEigenPolynomials(self, bandnum):
        '''
        Returns [ (xorder, yorder, coeffs),  (xorder, yorder, coeffs), ...]
        one tuple per eigen-PSF.
        xorder and yorder are np arrays of integers
        coeffs is a numpy array of floating-point coefficients
        '''
        T = fits_table(self.hdus[bandnum+1].data)
        terms = []
        for k in range(len(T)):
            nrb = T.nrow_b[k]
            ncb = T.ncol_b[k]
            c = T.c[k]
            # !!!
            c = c.copy()
            c = c.reshape(5, 5)
            c = c[:nrb,:ncb]
            (gridc,gridr) = np.meshgrid(np.arange(ncb), np.arange(nrb))
            # remove the 1e-3 coordinate prescaling
            c *= (1e-3 ** (gridr + gridc))
            I = np.flatnonzero(c)
            terms.append((gridr.flat[I], gridc.flat[I], c.flat[I]))
        return terms

    def correlateEigenPsf(self, bandnum, img):
        from scipy.ndimage.filters import correlate

        eigenpsfs = self.getEigenPsfs(bandnum)
        eigenterms = self.getEigenPolynomials(bandnum)
        H,W = img.shape
        corr = np.zeros((H,W))
        xx,yy = np.arange(W).astype(float), np.arange(H).astype(float)
        for epsf, (XO,YO,C) in zip(eigenpsfs, eigenterms):
            k = reduce(np.add, [np.outer(yy**yo, xx**xo) * c
                                for xo,yo,c in zip(XO,YO,C)])
            assert(k.shape == img.shape)
            # Trim symmetric zero-padding off the epsf.
            # This will fail spectacularly given an all-zero eigen-component.
            while True:
                H,W = epsf.shape
                if (np.all(epsf[:,0] == 0) and np.all(epsf[:,-1] == 0) and
                    np.all(epsf[0,:] == 0) and np.all(epsf[-1,:] == 0)):
                    # Trim!
                    epsf = epsf[1:-1, 1:-1]
                else:
                    break
            corr += k * correlate(img, epsf)
        return corr

    def getPsfAtPoints(self, bandnum, x, y):
        '''
        Reconstruct the SDSS model PSF from KL basis functions.

        x,y can be scalars or 1-d numpy arrays.

        Return value:
        if x,y are scalars: a PSF image
        if x,y are arrays:  a list of PSF images
        '''
        rtnscalar = np.isscalar(x) and np.isscalar(y)
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)

        eigenpsfs = self.getEigenPsfs(bandnum)
        eigenpolys = self.getEigenPolynomials(bandnum)

        # From the IDL docs:
        # http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
        #   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
        #   psfimage = SUM_k{ acoeff_k * RROWS_k }

        # we assume all the eigen-psfs are the same size.
        assert(len(np.unique([psf.shape for psf in eigenpsfs])) == 1)

        xx,yy = np.broadcast_arrays(x, y)
        N = len(xx.flat)
        psfimgs = np.zeros((N,) + eigenpsfs[0].shape)
        for epsf, (XO, YO, C) in zip(eigenpsfs, eigenpolys):
            kk = reduce(np.add, [(xx.flat ** xo) * (yy.flat ** yo) * c
                                 for (xo,yo,c) in zip(XO,YO,C)])
            psfimgs += epsf[np.newaxis,:,:] * kk[:,np.newaxis,np.newaxis]

        if rtnscalar:
            return psfimgs[0,:,:]
        # convert back to a list...
        return [psfimgs[i,:,:] for i in range(N)]

    def getGain(self, band=None):
        if band is not None:
            return self.gain[band]
        return self.gain

    def getDarkVariance(self, band=None):
        if band is not None:
            return self.dark_variance[band]
        return self.dark_variance

    def getSky(self, band=None):
        if band is not None:
            return self.sky[band]
        return self.sky

    def getSkyErr(self, band=None):
        if band is not None:
            return self.skyerr[band]
        return self.skyerr


