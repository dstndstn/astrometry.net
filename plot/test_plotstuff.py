# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import unittest

try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
import numpy as np
import pylab as plt
from math import pi,sqrt

from astrometry.plot.plotstuff import *

class TestPlotstuff(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    
    def test_image_wcs(self):
        '''
        Round-trip an image through a WCS.
        '''
        W,H = 100,100
        I = np.random.uniform(size=(H,W))
        imfn = 'test-plotstuff-1.fits'
        pyfits.writeto(imfn, I, clobber=True)

        wcs = anwcs_create_box(43., 11., 1., W, H)
        anwcs_rotate_wcs(wcs, 32.)
        wcsfn = 'test-plotstuff-1.wcs'
        anwcs_write(wcs, wcsfn)

        plot = Plotstuff()
        plot.outformat = PLOTSTUFF_FORMAT_PNG
        plot.size = (W, H)
        plot.wcs_file = wcsfn
        plot.color = 'black'
        plot.plot('fill')

        im = plot.image
        plot_image_set_wcs(im, wcsfn, 0)
        im.image_high = 1.
        im.image_low = 0.
        plot_image_set_filename(im, imfn)
        plot.plot('image')
        plotfn = 'test-plotstuff-1.png'
        plot.write(plotfn)

        I2 = plt.imread(plotfn)
        print(I2.shape)
        I2 = I2[:,:,0]

        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(I, vmin=0, vmax=1)
        plt.subplot(2,2,2)
        plt.imshow(I2, vmin=0, vmax=1)
        plt.subplot(2,2,3)
        plt.imshow(I2 - I, vmin=-1, vmax=1)
        plt.savefig('test-plotstuff-1b.png')

        md = np.max(np.abs((I2-I).ravel()))
        print('max diff', md)
        self.assertTrue(md < 1./255.)



    def test_subpixel_image_wcs(self):
        '''
        Create a well-sampled Gaussian test image, and push it through subpixel-shifted WCSes.
        '''
        W,H = 100,100
        X,Y = np.meshgrid(np.arange(W),np.arange(H))
        s = 2.
        CX,CY = 32.4,42.3
        I = 1./(2.*pi*s**2) * np.exp(((X-CX)**2 + (Y-CY)**2) / (-2*s**2))
        I /= I.max()

        print('I sum', I.sum())
        print('I max', I.max())
        print('first moment', np.mean(I * X) / np.mean(I), np.mean(I * Y) / np.mean(I))
        imfn = 'test-plotstuff-2.fits'
        pyfits.writeto(imfn, I, clobber=True)

        wcs = anwcs_create_box(43., 11., 1., W, H)
        anwcs_rotate_wcs(wcs, 32.)
        wcsfn = 'test-plotstuff-2.wcs'
        anwcs_write(wcs, wcsfn)

        plot = Plotstuff()
        plot.outformat = PLOTSTUFF_FORMAT_PNG
        plot.size = (W, H)
        plot.wcs_file = wcsfn
        plot.color = 'black'
        plot.plot('fill')

        im = plot.image
        plot_image_set_wcs(im, wcsfn, 0)
        im.image_high = I.max()
        im.image_low = 0.
        plot_image_set_filename(im, imfn)
        plot.plot('image')
        plotfn = 'test-plotstuff-2.png'
        plot.write(plotfn)

        I2 = plt.imread(plotfn)
        print(I2.shape)
        I2 = I2[:,:,0]

        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(I, vmin=0, vmax=I.max())
        plt.subplot(2,2,2)
        plt.imshow(I2, vmin=0, vmax=1)
        plt.subplot(2,2,3)
        plt.imshow(I2 - I, vmin=-1, vmax=1)
        plt.savefig('test-plotstuff-2b.png')

        md = np.max(np.abs((I2-I).ravel()))
        print('max diff', md)
        self.assertTrue(md < 1./255.)




        
if __name__ == '__main__':
    unittest.main()
       
