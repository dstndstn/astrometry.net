# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import os

from .common import *

from astrometry.util.miscutils import *
from astrometry.util.fits import *

class DR7(SdssDR):
    def __init__(self, **kwargs):
        '''
        kwargs:
        
        (base class:)
        curl=False: use curl rather than wget?
        basedir=None: base directory for local files
        '''
        SdssDR.__init__(self, **kwargs)
        # These are *LOCAL* filenames -- some are different than those
        # on the DAS.
        self.filenames = {
            'fpObjc': 'fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
            'fpM': 'fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
            'fpC': 'fpC-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
            'fpAtlas': 'fpAtlas-%(run)06i-%(camcol)i-%(field)04i.fit',
            'psField': 'psField-%(run)06i-%(camcol)i-%(field)04i.fit',
            #'tsObj': 'tsObj-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
            #'tsField': 'tsField-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
            'tsObj': 'tsObj-%(run)06i-%(camcol)i-%(field)04i.fit',
            'tsField': 'tsField-%(run)06i-%(camcol)i-%(field)04i.fit',
            }
        self.softbias = 1000

    def getDRNumber(self):
        return 7
        
    def retrieve(self, filetype, run, camcol, field, band=None, skipExisting=True):
        # FIXME!
        from astrometry.util.sdss_das import sdss_das_get
        outfn = self.getPath(filetype, run, camcol, field, band)
        #print 'Output filename:', outfn
        if skipExisting and os.path.exists(outfn):
            return
        return sdss_das_get(filetype, outfn, run, camcol, field, band,
                            curl=self.curl)

    def readTsField(self, run, camcol, field, rerun):
        '''
        http://www.sdss.org/dr7/dm/flatFiles/tsField.html

        band: string ('u', 'g', 'r', 'i', 'z')
        '''
        f = TsField(run, camcol, field, rerun=rerun)
        fn = self.getFilename('tsField', run, camcol, field, rerun=rerun)
        #print 'reading file', fn
        p = self._open(fn)
        #print 'got', len(p), 'HDUs'
        f.setHdus(p)
        return f

    def readFpC(self, run, camcol, field, band):
        '''
        http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpC.html

        band: string ('u', 'g', 'r', 'i', 'z')
        '''
        f = FpC(run, camcol, field, band)
        # ...
        fn = self.getFilename('fpC', run, camcol, field, band)
        #print 'reading file', fn
        p = self._open(fn)
        #print 'got', len(p), 'HDUs'
        f.image = p[0].data
        f.header = p[0].header
        return f

    def readFpObjc(self, run, camcol, field):
        '''
        http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpObjc.html
        '''
        f = FpObjc(run, camcol, field)
        # ...
        fn = self.getFilename('fpObjc', run, camcol, field)
        #print 'reading file', fn
        p = self._open(fn)
        #print 'got', len(p), 'HDUs'
        return f

    def readFpM(self, run, camcol, field, band):
        '''
        http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
        '''
        f = FpM(run, camcol, field, band)
        # ...
        fn = self.getFilename('fpM', run, camcol, field, band)
        #print 'reading file', fn
        p = self._open(fn)
        #print 'got', len(p), 'HDUs'
        f.setHdus(p)
        return f

    def readPsField(self, run, camcol, field):
        '''
        http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
        '''
        f = PsField(run, camcol, field)
        # ...
        fn = self.getFilename('psField', run, camcol, field)
        #print 'reading file', fn
        p = self._open(fn)
        #print 'got', len(p), 'HDUs'
        f.setHdus(p)
        return f

    def getInvvar(self, fpC, fpM, gain, darkvar, sky, skyerr,
                  x0=0, x1=None, y0=0, y1=None, invvar_and_mask=False):
        '''
        Produces a (cut-out of) the inverse-variance noise image, from columns
        [x0,x1] and rows [y0,y1] (inclusive).  Default is the whole image.

        fpC is the image pixels (eg FpC.getImage())
        #### CHECK THIS -- below we have  (img + sky), but fpCs have *not*
        had sky subtracted.

        fpM is the FpM
        gain, darkvar, sky, and skyerr can be retrieved from the psField file.
        '''
        if x1 is None:
            x1 = fpC.shape[1]-1
        if y1 is None:
            y1 = fpC.shape[0]-1

        # Poisson: mean = variance
        # Add readout noise?
        # Spatial smoothing?
        img = fpC[y0:y1+1, x0:x1+1]

        # from http://www.sdss.org/dr7/algorithms/fluxcal.html
        ivarimg = 1./((img + sky) / gain + darkvar + skyerr)

        if invvar_and_mask:
            mask = np.ones(ivarimg.shape, bool)
            maskimg = mask
        else:
            maskimg = ivarimg
        
        # Noise model:
        #  -mask coordinates are wrt fpC coordinates.
        #  -INTERP, SATUR, CR,
        #  -GHOST?
        for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
            fpM.setMaskedPixels(plane, maskimg, 0)

        if invvar_and_mask:
            return ivarimg, mask
        return ivarimg


