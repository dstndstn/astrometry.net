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
from astrometry.util.fits import *
from astrometry.plot.plotstuff import *
import numpy as np
import pylab as plt
from math import pi,sqrt

class TestPlotstuff2(unittest.TestCase):

    def testSubpixelShift(self):
        W,H = 100,100
        cx,cy = 34.2, 56.7
        X,Y = np.meshgrid(np.arange(W), np.arange(H))
        S = 2.
        G = 1./(2.*pi*S**2) * np.exp(-((X-cx)**2+(Y-cy)**2)/(2.*S**2))
        #G /= G.max()
        print(G.sum())
        print(np.sum(G*X), np.sum(G*Y))
        imfn = 'test-plotstuff-2.fits'
        pyfits.writeto(imfn, G, clobber=True)

        wcs = anwcs_create_box(33.5, 10.4, 1., W, H)
        anwcs_rotate_wcs(wcs, 25.)
        wcsfn = 'test-plotstuff-2.wcs'
        anwcs_write(wcs, wcsfn)

        plot = Plotstuff()
        plot.outformat = PLOTSTUFF_FORMAT_PNG
        plot.size = (W,H)
        plot.wcs_file = wcsfn
        plot.color = 'black'
        plot.plot('fill')

        im = plot.image
        im.image_low = 0
        im.image_high = G.max()
        plot_image_set_wcs(im, wcsfn, 0)
        plot_image_set_filename(im, imfn)
        plot.plot('image')

        plotfn = 'test-plotstuff-2.png'
        plot.write(plotfn)

        I = plt.imread(plotfn)
        I = I[:,:,0]
        sx,sy = (I*X).sum()/I.sum(), (I*Y).sum()/I.sum()
        print(sx,sy)
        ex,ey = cx,cy
        self.assertTrue(abs(sx-ex) < 0.1)
        self.assertTrue(abs(sy-ey)< 0.1)

        # Shift the plot's WCS CRPIX.
        dx = 0.25
        dy = 0.3
        sip = anwcs_get_sip(wcs)
        tan = sip.wcstan
        plotstuff_set_wcs(plot.pargs, wcs)

        xy = plot.xy
        plot_xy_set_wcs_filename(xy, wcsfn)
        plot.marker = 'crosshair'

        # Move the plot WCS origin
        for step in range(16):
            if step < 8:
                tan.set_crpix(tan.crpix[0]+dx, tan.crpix[1])
                ex += dx
            else:
                tan.set_crpix(tan.crpix[0], tan.crpix[1]+dy)
                ey += dy
            #anwcs_print_stdout(wcs)
            plot.color = 'black'
            plot.plot('fill')
            plot.plot('image')
            plotfn = 'test-plotstuff-2b%i.png' % step
            plot.write(plotfn)
            #anwcs_write(plot.pargs.wcs, 'test-plotstuff-2b.wcs')
            I = plt.imread(plotfn)
            I = I[:,:,0]
            sx,sy = (I*X).sum()/I.sum(), (I*Y).sum()/I.sum()
            print(sx,sy)
            print(ex,ey)
            self.assertTrue(abs(sx-ex) < 0.1)
            self.assertTrue(abs(sy-ey) < 0.1)

            plot.color = 'red'
            plot_xy_clear_list(xy)
            # don't plot at sx,sy / ex,ey -- original image coords
            # are unchanged.
            plot_xy_vals(xy, cx+1, cy+1)
            plot.plot('xy')
            plotfn = 'test-plotstuff-2c%i.png' % step
            plot.write(plotfn)
            # visual check that plot xy symbols match source centroid -- yes
            

        # Scan image WCS in RA,Dec; check that recovered source position
        # through plot WCS matches.
        # reset...
        plot.wcs_file = wcsfn
        plotwcs = anwcs_open(wcsfn, 0)

        wcs = anwcs_open(wcsfn, 0)
        im.wcs = wcs
        sip = anwcs_get_sip(wcs)
        tan = sip.wcstan
        ddec = 1.2 * 1./W
        ok,era,edec = anwcs_pixelxy2radec(wcs, cx, cy)
        print(era,edec)
        for step in range(16):
            tan.set_crval(tan.crval[0], tan.crval[1] + ddec)
            edec += ddec

            plot.color = 'black'
            plot.plot('fill')
            plot.plot('image')
            plotfn = 'test-plotstuff-2d%i.png' % step
            plot.write(plotfn)

            I = plt.imread(plotfn)
            I = I[:,:,0]
            sx,sy = (I*X).sum()/I.sum(), (I*Y).sum()/I.sum()
            #print sx,sy
            ok,ra,dec = anwcs_pixelxy2radec(plotwcs, sx, sy)

            #print era,edec
            #print ra,dec
            print('dRA,dDec', ra-era, dec-edec)
            self.assertTrue(abs(ra-era) < 1e-4)
            self.assertTrue(abs(dec-edec) < 1e-4)

if __name__ == '__main__':
    unittest.main()
