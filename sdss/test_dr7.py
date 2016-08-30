# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import os
from math import sqrt

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import pylab as plt

import astrometry.sdss
from astrometry.sdss import DR7
from astrometry.util.fits import fits_table

if __name__ == '__main__':
    sdss = DR7()

    testdata = os.path.join(os.path.dirname(astrometry.sdss.__file__),
                            'testdata')
    print('Using test data dir:', testdata)

    sdss.setBasedir(testdata)

    tsfield = sdss.readTsField(2830, 6, 398, 41)

    rband,iband = 2,3

    asr = tsfield.getAsTrans(rband)
    asi = tsfield.getAsTrans(iband)

    '''
$ listhead fpC-002830-r6-0398.fit

CRPIX1  = 1.02450000000000E+03 / Column Pixel Coordinate of Ref. Pixel
CRPIX2  = 7.44500000000000E+02 / Row Pixel Coordinate of Ref. Pixel
CRVAL1  = 1.79464261370000E+02 / RA at Reference Pixel
CRVAL2  = 5.33206072700000E+01 / DEC at Reference Pixel
CD1_1   = -8.4291038759718E-06 / RA  degrees per column pixel
CD1_2   = 1.09673333339369E-04 / RA  degrees per row pixel
CD2_1   = 1.09675673828125E-04 / DEC degrees per column pixel
CD2_2   = 8.42453629032368E-06 / DEC degrees per row pixel
'''

    print('CD at 0,0:', asr.cd_at_pixel(1024.5, 744.5))

    for x,y,color in [ (0, 0, 0),

                       (np.array([1,2,3]),
                        np.array([0,100,200]),
                        0),

                       (np.array([1,2,3]),
                        np.array([0,100,200]),
                        np.array([0,1,2])),
                       ]:
        print()
        print('Pixel x,y', x,y)
        print('color', color)
        rr,dr = asr.pixel_to_radec(x, y, color)
        ri,di = asi.pixel_to_radec(x, y, color)
        print('r-band RA,Dec:', rr,dr)
        print('i-band RA,Dec:', ri,di)
        rx,ry = asr.radec_to_pixel(rr, dr)
        ix,iy = asi.radec_to_pixel(ri, di)
        print('r-band x,y:', rx, ry)
        print('i-band x,y:', ix, iy)


    tsobj = fits_table(os.path.join(testdata, 'cut-tsObj-002830-6-0-0398.fit'))
    ra,dec = tsobj.ra, tsobj.dec
    #X,Y = tsobj.colc[:,rband], tsobj.rowc[:,rband]
    X,Y = tsobj.objc_colc, tsobj.objc_rowc
    print('ra,dec', ra.shape, dec.shape)
    print('x', X.shape)
    #rmag = tsobj.psfcounts[:,rband]
    #imag = tsobj.psfcounts[:,iband]
    rmag = tsobj.counts_model[:,rband]
    imag = tsobj.counts_model[:,iband]
    # According to http://www.sdss.org/dr7/dm/flatFiles/tsField.html,
    #   r,i,z fields, use this color:
    color = (rmag - imag)
    print('color:', color.min(), color.max())

    xx,yy = asr.radec_to_pixel(ra, dec, color)
    print('xx', xx.shape)
    rr,dd = asr.pixel_to_radec(X, Y, color)

    print('dxy', xx-X, yy-Y)
    #print 'dradec', ra-rr, dec-dd

    print('x RMS:', sqrt(np.mean((xx - X)**2)))
    print('y RMS:', sqrt(np.mean((yy - Y)**2)))
    print('RA RMS:', sqrt(np.mean((ra - rr)**2)))
    print('Dec RMS:', sqrt(np.mean((dec - dd)**2)))

    #I = np.argmax((xx - X)**2 + (yy - Y)**2)
    #print 'Biggest x,y deviant: row', I
    #tsobj[I].about()
    
    plt.clf()
    plt.plot(xx,yy, 'r.')
    plt.plot(X, Y,  'bo', mfc='none')
    plt.savefig('dxy.png')
    
    lo,hi = 1e-10, 10
    rng = np.log10(lo), np.log10(hi)
    plt.clf()
    plt.hist(np.log10(np.clip(np.abs(xx - X), lo, hi)), range=rng)
    plt.xlabel('log_10 ( x error )')
    plt.savefig('dx.png')

    plt.clf()
    plt.hist(np.log10(np.clip(np.abs(yy - Y), lo, hi)), range=rng)
    plt.xlabel('log_10 ( y error )')
    plt.savefig('dy.png')

    plt.clf()
    plt.hist(np.log10(np.clip(np.sqrt((yy - Y)**2 + (xx - X)**2), lo, hi)),
             range=rng)
    plt.xlabel('log_10 ( x,y error )')
    plt.savefig('dr.png')

