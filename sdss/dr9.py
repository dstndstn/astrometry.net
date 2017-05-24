# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import
from .common import *
from .dr8 import *

class DR9(DR8):

    def __init__(self, **kwargs):
        '''
        Useful kwargs:
        
        basedir : (string) - local directory where data will be stored.
        '''
        DR8.__init__(self, **kwargs)
        self.dasurl = 'http://data.sdss3.org/sas/dr9/boss/'

    def getDRNumber(self):
        return 9
        
    def _get_runlist_filename(self):
        return self._get_data_file('runList-dr9.par')


if __name__ == '__main__':
    sdss = DR9()
    rcfb = (2873, 3, 211, 'r')
    r,c,f,b = rcfb
    bandnum = band_index(b)
    sdss.retrieve('psField', *rcfb)
    psfield = sdss.readPsField(r,c,f)
    dg = psfield.getDoubleGaussian(bandnum, normalize=True)

    psf = psfield.getPsfAtPoints(bandnum, 2048/2., 1489./2.)

    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    import numpy as np
    
    H,W = psf.shape
    cx,cy = (W/2, H/2)
    DX,DY = np.meshgrid(np.arange(W)-cx, np.arange(H)-cy)
    (a1,s1, a2,s2) = dg
    R2 = (DX**2 + DY**2)
    G = (a1 / (2.*np.pi*s1**2) * np.exp(-R2/(2.*s1**2)) +
         a2 / (2.*np.pi*s2**2) * np.exp(-R2/(2.*s2**2)))
    print('G sum', G.sum())
    print('psf sum', psf.sum())
    psf /= psf.sum()
    
    plt.clf()
    plt.subplot(2,2,1)
    ima = dict(interpolation='nearest', origin='lower')
    plt.imshow(psf, **ima)
    plt.subplot(2,2,2)
    plt.imshow(G, **ima)

    plt.subplot(2,2,3)
    plt.plot(psf[H/2,:], 'rs-', mec='r', mfc='none')
    plt.plot(G[H/2,:], 'gx-')

    plt.subplot(2,2,4)
    plt.semilogy(np.maximum(1e-6, psf[H/2,:]), 's-', mec='r', mfc='none')
    plt.semilogy(np.maximum(1e-6, G[H/2,:]), 'gx-')

    plt.savefig('psf1.png')


    
    
