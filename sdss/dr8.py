# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import
import os
from astrometry.util.fits import fits_table
import numpy as np
import logging
import tempfile

import sys
py3 = (sys.version_info[0] >= 3)
if py3:
    from urllib.parse import urljoin
else:
    from urlparse import urljoin

fitsio = None
try:
    import fitsio
except:
    try:
        import pyfits
    except ImportError:
        try:
            from astropy.io import fits as pyfits
        except ImportError:
            raise ImportError("Cannot import either pyfits or astropy.io.fits")

from .common import *
from .dr7 import *
from .yanny import *
from astrometry.util.run_command import run_command

class Frame(SdssFile):
    def __init__(self, *args, **kwargs):
        super(Frame, self).__init__(*args, **kwargs)
        self.filetype = 'frame'
        self.image = None

        self.image_proxy = None

    def getImageShape(self):
        if self.image_proxy is not None:
            # fitsio fits.FITSHDU object
            H,W = self.image_proxy.get_info()['dims']
            H = int(H)
            W = int(W)
        else:
            H,W = self.image.shape
        return H,W

    def getImageSlice(self, slice):
        if self.image_proxy is not None:
            #print 'reading slice from image proxy:', slice
            return self.image_proxy[slice]
        return self.image[slice]

    #def __str__(self):
    def getImage(self):
        if self.image is None and self.image_proxy is not None:
            self.image = self.image_proxy.read()
            self.image_proxy = None
        return self.image
    def getHeader(self):
        return self.header
    def getAsTrans(self):
        return self.astrans
    def getCalibVec(self):
        return self.calib

    def getSkyAt(self, x, y):
        skyim = self.sky
        (sh,sw) = skyim.shape
        if sw != 256:
            skyim = skyim.T
        (sh,sw) = skyim.shape
        xi = np.round(self.skyxi[x]).astype(int)
        yi = np.round(self.skyyi[y]).astype(int)
        yi = np.minimum(yi,sh-1)
        return skyim[yi,xi]
    
    def getSky(self):
        skyim = self.sky
        (sh,sw) = skyim.shape
        if sw != 256:
            skyim = skyim.T
        (sh,sw) = skyim.shape
        xi = np.round(self.skyxi).astype(int)
        yi = np.round(self.skyyi).astype(int)
        yi = np.minimum(yi,sh-1)
        assert(all(xi >= 0) and all(xi < sw))
        assert(all(yi >= 0) and all(yi < sh))
        XI,YI = np.meshgrid(xi, yi)
        # Nearest-neighbour interpolation -- we just need this
        # for approximate invvar.
        bigsky = skyim[YI,XI]
        return bigsky

    def getInvvar(self, psfield, bandnum, ignoreSourceFlux=False,
                  sourceFlux=None, constantSkyAt=None):
        '''
        If constantSkyAt = (x,y) (INTEGERS!),
        returns a scalar (rather than a np.array) of the invvar at that point.
        
        NOTE that this does NOT blank out masked pixels; use, eg,

        fpM = sdss.readFpM(run, camcol, field, bandname)
        for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
            fpM.setMaskedPixels(plane, invvar, 0, roi=roi)
        '''
        calibvec = self.getCalibVec()

        if constantSkyAt:
            x,y = constantSkyAt
            calibvec = calibvec[x]
            sky = self.getSkyAt(x,y)
            if ignoreSourceFlux:
                dn = sky
            elif sourceFlux is None:
                image = self.getImage()
                dn = (image[y,x] / calibvec) + sky
            else:
                dn = (sourceFlux / calibvec) + sky
        else:
            bigsky = self.getSky()
            if ignoreSourceFlux:
                dn = bigsky
            elif sourceFlux is None:
                image = self.getImage()
                dn = (image / calibvec) + bigsky
            else:
                dn = (sourceFlux / calibvec) + bigsky

        gain = psfield.getGain(bandnum)
        # Note, "darkvar" includes dark current *and* read noise.
        darkvar = psfield.getDarkVariance(bandnum)
        dnvar = (dn / gain) + darkvar
        invvar = 1./(dnvar * calibvec**2)
        return invvar

class PhotoObj(SdssFile):
    def __init__(self, *args, **kwargs):
        super(PhotoObj, self).__init__(*args, **kwargs)
        self.filetype = 'photoObj'
        self.table = None
    def getTable(self):
        return self.table

class runlist(object):
    pass

class DR8(DR7):
    _lup_to_mag_b = np.array([1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10])
    _two_lup_to_mag_b = 2.*_lup_to_mag_b
    _ln_lup_to_mag_b = np.log(_lup_to_mag_b)

    '''
    From
    http://data.sdss3.org/datamodel/glossary.html#asinh

    m = -(2.5/ln(10))*[asinh(f/2b)+ln(b)].

    The parameter b is a softening parameter measured in maggies, and
    for the [u, g, r, i, z] bands has the values
    [1.4, 0.9, 1.2, 1.8, 7.4] x 1e-10
    '''
    @staticmethod
    def luptitude_to_mag(Lmag, bandnum, badmag=25):
        if bandnum is None:
            # assume Lmag is broadcastable to a 5-vector
            twobi = DR8._two_lup_to_mag_b
            lnbi = DR8._ln_lup_to_mag_b
        else:
            twobi = DR8._two_lup_to_mag_b[bandnum]
            lnbi = DR8._ln_lup_to_mag_b[bandnum]
        # MAGIC -1.08.... = -2.5/np.log(10.)
        f = np.sinh(Lmag/-1.0857362047581294 - lnbi) * twobi
        # prevent log10(-flux)
        mag = np.zeros_like(f) + badmag
        I = (f > 0)
        mag[I] = -2.5 * np.log10(f[I])
        return mag

    @staticmethod
    def nmgy_to_mag(nmgy):
        return 22.5 - 2.5 * np.log10(nmgy)

    def getDRNumber(self):
        return 8

    def useLocalTree(self, photoObjs=None, resolve=None):
        if photoObjs is None:
            photoObjs = os.environ['BOSS_PHOTOOBJ']
        redux = os.environ['PHOTO_REDUX']
        if resolve is None:
            resolve = os.environ['PHOTO_RESOLVE']
        
        self.filenames.update(
            photoObj = os.path.join(photoObjs, '%(rerun)s', '%(run)i', '%(camcol)i',
                                    'photoObj-%(run)06i-%(camcol)i-%(field)04i.fits'),
            frame = os.path.join(photoObjs, 'frames', '%(rerun)s', '%(run)i', '%(camcol)i',
                                    'frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits.bz2'),
            photoField = os.path.join(photoObjs, '%(rerun)s', '%(run)i',
                                      'photoField-%(run)06i-%(camcol)i.fits'),
            psField = os.path.join(redux, '%(rerun)s', '%(run)i', 'objcs', '%(camcol)i',
                                   'psField-%(run)06i-%(camcol)i-%(field)04i.fit'),
            fpM = os.path.join(redux, '%(rerun)s', '%(run)i', 'objcs', '%(camcol)i',
                               'fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit.gz'),
            window_flist = os.path.join(resolve, 'window_flist.fits'),
            )
        # use fpM files compressed
        try:
            del self.dassuffix['fpM']
        except:
            pass
        try:
            del self.processcmds['fpM']
        except:
            pass

    def saveUnzippedFiles(self, basedir):
        self.unzip_dir = basedir

    def setFitsioReadBZ2(self, to=True):
        '''
        Call this if fitsio supports reading .bz2 files directly.
        '''
        self.readBz2 = to
    
    def __init__(self, **kwargs):
        '''
        Useful kwargs:
        
        basedir : (string) - local directory where data will be stored.
        '''
        DR7.__init__(self, **kwargs)

        self.unzip_dir = None
        self.readBz2 = False

        # Local filenames
        self.filenames.update({
            'frame': 'frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits.bz2',
            'idR': 'idR-%(run)06i-%(band)s-%(camcol)i-%(field)04i.fits',
            'photoObj': 'photoObj-%(run)06i-%(camcol)i-%(field)04i.fits',
            'photoField': 'photoField-%(run)06i-%(camcol)i.fits',
            'window_flist': 'window_flist.fits',
            })

        # URLs on DAS server
        self.dasurl = 'http://data.sdss3.org/sas/dr8/groups/boss/'
        self.daspaths = {
            'idR': 'photo/data/%(run)i/fields/%(camcol)i/idR-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit.Z',
            'fpObjc': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
            # DR8 frames are no longer available on DAS.
            'frame': '/sas/dr9/boss/photoObj/frames/%(rerun)s/%(run)i/%(camcol)i/frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits.bz2',
            #'frame': 'photoObj/frames/%(rerun)s/%(run)i/%(camcol)i/frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits.bz2',
            'photoObj': 'photoObj/%(rerun)s/%(run)i/%(camcol)i/photoObj-%(run)06i-%(camcol)i-%(field)04i.fits',
            'psField': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/psField-%(run)06i-%(camcol)i-%(field)04i.fit',
            'photoField': 'photoObj/%(rerun)s/%(run)i/photoField-%(run)06i-%(camcol)i.fits',
            'fpM': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit.gz',
            'fpAtlas': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/fpAtlas-%(run)06i-%(camcol)i-%(field)04i.fit',
            'window_flist': 'resolve/2010-05-23/window_flist.fits',
            }

        self.dassuffix = {
            #'frame': '.bz2',
            'fpM': '.gz',
            'idR': '.Z',
            }

        # called in retrieve()
        self.processcmds = {
            'fpM': 'gunzip -cd %(input)s > %(output)s',
            'idR': 'gunzip -cd %(input)s > %(output)s',
            }

        self.postprocesscmds = {
            'frame': 'TMPFILE=$(mktemp %(output)s.tmp.XXXXXX) && bunzip2 -cd %(input)s > $TMPFILE && mv $TMPFILE %(output)s',
            }
        
        y = read_yanny(self._get_runlist_filename())
        y = y['RUNDATA']
        rl = runlist()
        rl.run = np.array(y['run'])
        rl.startfield = np.array(y['startfield'])
        rl.endfield = np.array(y['endfield'])
        rl.rerun = np.array(y['rerun'])
        #print 'Rerun type:', type(rl.rerun), rl.rerun.dtype
        self.runlist = rl

        self.logger = logging.getLogger('astrometry.sdss.DR%i' %
                                        self.getDRNumber())
        #self.logger.debug('debug test')
        #self.logger.info('info test')
        #self.logger.warning('warning test')

    def _unzip_frame(self, fn, run, camcol):
        if self.readBz2:
            return None,True

        # No, PJM reported that pyfits failed on SDSS frame*.bz2 files
        # if not fitsio:
        #    # pyfits can read .bz2
        #    return None,True

        tempfn = None
        keep = False

        filetype = 'frame'
        if not(filetype in self.postprocesscmds and fn.endswith('.bz2')):
            return None,True
        
        cmd = self.postprocesscmds[filetype]

        if self.unzip_dir is not None:
            udir = os.path.join(self.unzip_dir, '%i' % run, '%i' % camcol)
            if not os.path.exists(udir):
                try:
                    os.makedirs(udir)
                except:
                    pass
            tempfn = os.path.join(udir, os.path.basename(fn).replace('.bz2', ''))
            #print 'Checking', tempfn
            if os.path.exists(tempfn):
                print('File exists:', tempfn)
                return tempfn,True
            else:
                print('Saving to', tempfn)
                keep = True

        else:
            fid,tempfn = tempfile.mkstemp()
            os.close(fid)

        cmd = cmd % dict(input = fn, output = tempfn)
        self.logger.debug('cmd: %s' % cmd)
        print('command:', cmd)
        (rtn,out,err) = run_command(cmd)
        if rtn:
            print('Command failed: command', cmd)
            print('Output:', out)
            print('Error:', err)
            print('Return val:', rtn)
            raise RuntimeError('Command failed (return val %i): %s' % (rtn, cmd))

        print(out)
        print(err)
        return tempfn,keep


        
        
    def _get_runlist_filename(self):
        return self._get_data_file('runList-dr8.par')

    # read a data file describing the DR8 data
    def _get_data_file(self, fn):
        return os.path.join(os.path.dirname(__file__), fn)

    def get_rerun(self, run, field=None):
        I = (self.runlist.run == run)
        if field is not None:
            I *= (self.runlist.startfield <= field) * (self.runlist.endfield >= field)
        I = np.flatnonzero(I)
        reruns = np.unique(self.runlist.rerun[I])
        #print 'Run', run, '-> reruns:', reruns
        if len(reruns) == 0:
            return None
        return reruns[-1]

    def get_url(self, filetype, run, camcol, field, band=None, rerun=None):
        if rerun is None:
            rerun = self.get_rerun(run, field)
        path = self.daspaths[filetype]
        url = urljoin(self.dasurl, path % dict(
            run=run, camcol=camcol, field=field, rerun=rerun, band=band))
        return url
    
    def retrieve(self, filetype, run, camcol, field=None, band=None, skipExisting=True,
                 tempsuffix='.tmp', rerun=None):

        outfn = self.getPath(filetype, run, camcol, field, band,
                             rerun=rerun)
        print('Checking for file', outfn)
        if outfn is None:
            return None
        if skipExisting and os.path.exists(outfn):
            #print('Exists')
            return outfn
        outdir = os.path.dirname(outfn)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        url = self.get_url(filetype, run, camcol, field, band=band, rerun=rerun)
        #print 'Did not find file:', outfn
        print('Retrieving from URL:', url)
        if self.curl:
            cmd = "curl -o '%(outfn)s' '%(url)s'"
        else:
            cmd = "wget --continue -nv -O %(outfn)s '%(url)s'"

        # suffix to add to the downloaded filename
        suff = self.dassuffix.get(filetype, '')

        oo = outfn + suff
        if tempsuffix is not None:
            oo += tempsuffix
        
        cmd = cmd % dict(outfn=oo, url=url)
        self.logger.debug('cmd: %s' % cmd)
        (rtn,out,err) = run_command(cmd)
        if rtn:
            print('Command failed: command', cmd)
            print('Output:', out)
            print('Error:', err)
            print('Return val:', rtn)
            return None

        if tempsuffix is not None:
            #
            self.logger.debug('Renaming %s to %s' % (oo, outfn+suff))
            os.rename(oo, outfn + suff)

        if filetype in self.processcmds:
            cmd = self.processcmds[filetype]
            cmd = cmd % dict(input = outfn + suff, output = outfn)
            self.logger.debug('cmd: %s' % cmd)
            (rtn,out,err) = run_command(cmd)
            if rtn:
                print('Command failed: command', cmd)
                print('Output:', out)
                print('Error:', err)
                print('Return val:', rtn)
                return None

        return outfn

    def readPhotoObj(self, run, camcol, field, filename=None):
        obj = PhotoObj(run, camcol, field)
        if filename is None:
            fn = self.getPath('photoObj', run, camcol, field)
        else:
            fn = filename
        obj.table = fits_table(fn)
        return obj

    def readFrame(self, run, camcol, field, band, filename=None):
        '''
        http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
        '''
        f = Frame(run, camcol, field, band)
        # ...
        if filename is None:
            fn = self.getPath('frame', run, camcol, field, band)
        else:
            fn = filename

        # optionally bunzip2 the frame file.
        tempfn,keep = self._unzip_frame(fn, run, camcol)
        if tempfn is not None:
            fn = tempfn

        if fitsio:

            print('Frame filename', fn)
            # eg /clusterfs/riemann/raid006/dr10/boss/photoObj/frames/301/2825/1/frame-u-002825-1-0126.fits.bz2

            F = fitsio.FITS(fn, lower=True)
            f.header = F[0].read_header()
            # Allow later reading of just the pixels of interest.
            f.image_proxy = F[0]
            f.calib = F[1].read()
            sky = F[2].read_columns(['allsky', 'xinterp', 'yinterp'])
            #print 'sky', type(sky)
            # ... supposed to be a recarray, but it's not...
            f.sky, f.skyxi, f.skyyi = sky.tolist()[0]
            tab = fits_table(F[3].read())
            if not keep and tempfn is not None:
                os.remove(tempfn)

        else:
            p = pyfits.open(fn)
            # in nanomaggies
            f.image = p[0].data
            f.header = p[0].header
            # converts counts -> nanomaggies
            f.calib = p[1].data
            # table with val,x,y -- binned; use bilinear interpolation to expand
            sky = p[2].data
            # table -- asTrans structure
            tab = fits_table(p[3].data)

            f.sky = sky.field('allsky')[0]
            f.skyxi = sky.field('xinterp')[0]
            f.skyyi = sky.field('yinterp')[0]

        #print 'sky shape', f.sky.shape
        if len(f.sky.shape) != 2:
            f.sky = f.sky.reshape((-1, 256))
        assert(len(tab) == 1)
        tab = tab[0]
        # DR7 has NODE, INCL in radians...
        f.astrans = AsTrans(run, camcol, field, band,
                            node=np.deg2rad(tab.node), incl=np.deg2rad(tab.incl),
                            astrans=tab, cut_to_band=False)
                            
        return f
    
