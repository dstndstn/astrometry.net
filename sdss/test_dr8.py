# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import sys
from astrometry.sdss.dr8 import *
import numpy as np

def test_astrans(sdss, r,c,f,b):
    bandnum = band_index(b)
    sdss.retrieve('frame', r, c, f, b)
    frame = sdss.readFrame(r, c, f, b)
    astrans = frame.getAsTrans()
    sdss.retrieve('photoObj', r, c, f)
    obj = sdss.readPhotoObj(r, c, f)
    tab = obj.getTable()
    #tab.about()
    x,y = tab.colc[:,bandnum], tab.rowc[:,bandnum]
    ra,dec = tab.ra, tab.dec


    for r,d in zip(ra,dec):
        print('ra,dec', r,d)
        #print 'py:'
        x1,y1 = astrans.radec_to_pixel_single_py(r, d)
        print('  py', x1,y1)
        #print 'c:'
        x2,y2 = astrans.radec_to_pixel_single_c(r, d)
        print('  c', x2,y2)
        assert(np.abs(x1 - x2) < 1e-6)
        assert(np.abs(y1 - y2) < 1e-6)




    r2,d2 = astrans.pixel_to_radec(x, y)
    plt.clf()
    plt.plot(ra, dec, 'r.')
    plt.plot(r2, d2, 'bo', mec='b', mfc='none')
    plt.savefig('rd.png')

    r3,d3 = [],[]
    for xi,yi in zip(x,y):
        ri,di = astrans.pixel_to_radec(xi, yi)
        r3.append(ri)
        d3.append(di)
    plt.clf()
    plt.plot(ra, dec, 'r.')
    plt.plot(r3, d3, 'bo', mec='b', mfc='none')
    plt.savefig('rd3.png')

    x2,y2 = astrans.radec_to_pixel(ra, dec)
    plt.clf()
    plt.plot(x, y, 'r.')
    plt.plot(x2, y2, 'bo', mec='b', mfc='none')
    plt.savefig('xy.png')

    x3,y3 = [],[]
    for ri,di in zip(ra, dec):
        xi,yi = astrans.radec_to_pixel(ri, di)
        x3.append(xi)
        y3.append(yi)
    plt.clf()
    plt.plot(x, y, 'r.')
    plt.plot(x3, y3, 'bo', mec='b', mfc='none')
    plt.savefig('xy3.png')


if __name__ == '__main__':
    sdss = DR8()
    #test_astrans(sdss, 4623, 1, 203, 'r')
    test_astrans(sdss, 5065, 1, 68, 'r')
    sys.exit(0)

    fnew  = sdss.readFrame(4623, 1, 203, 'r', filename='frame-r-004623-1-0203.fits')
    print('fnew:', fnew)
    forig = sdss.readFrame(4623, 1, 203, 'r', 'frame-r-004623-1-0203.fits.orig')
    print('forig:', forig)

    frame = sdss.readFrame(3712, 3, 187, 'r')
    print('frame:', frame)
    img = frame.getImage()
    print('  image', img.shape)

    fpobj = sdss.readFpObjc(6581, 2, 135)
    print('fpobj:', fpobj)

    fpm = sdss.readFpM(6581, 2, 135, 'i')
    print('fpm:', fpm)

    psf = sdss.readPsField(6581, 2, 135)
    print('psfield:', psf)


    
