# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import sys

import matplotlib
matplotlib.use('Agg')

from math import pi
from pylab import *
from numpy import *
from numpy.random import *
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")

# Given an image and an xylist (including estimated image sigma),
# look at a cutout around each source position, add noise, and recompute
# dcen3x3 to find the noisy peak.

def dcen3a(f0, f1, f2):
    s = 0.5 * (f2 - f0)
    d = 2. * f1 - (f0 + f2)
    if (d <= 1.e-10*f0):
        return None
    aa = f1 + 0.5 * s * s / d
    sod = s / d
    return sod * (1. + (4./3.) * (0.25 * d / aa) * (1. - 4. * sod * sod)) + 1.


def dcen3b(f0, f1, f2):
    a = 0.5 * (f2 - 2*f1 + f0)
    b = f1 - a - f0
    xc = -0.5 * b / a
    if not (0.0 < xc < 2.0):
        return None
    return xc

def dcen3(f0, f1, f2):
    return dcen3a(f0,f1,f2)

def dcen3x3(image):
    my0 = dcen3(image[0,0], image[1,0], image[2,0])
    my1 = dcen3(image[0,1], image[1,1], image[2,1])
    my2 = dcen3(image[0,2], image[1,2], image[2,2])

    mx0 = dcen3(image[0,0], image[0,1], image[0,2])
    mx1 = dcen3(image[1,0], image[1,1], image[1,2])
    mx2 = dcen3(image[2,0], image[2,1], image[2,2])

    if None in [ mx0, mx1, mx2, my0, my1, my2 ]:
        return None

    # x = (y-1) mx + bx
    bx = (mx0 + mx1 + mx2) / 3.
    mx = (mx2 - mx0) / 2.

    # y = (x-1) my + by
    by = (my0 + my1 + my2) / 3.
    my = (my2 - my0) / 2.;

    # find intersection
    xc = (mx * (by - my - 1.) + bx) / (1. + mx * my)
    yc = (xc - 1.) * my + by

    # check that we are in the box
    if not ((0.0 < xc < 2.0) and (0.0 < yc < 2.0)):
        return None

    return (xc,yc)


if __name__ == '__main__':
    imgfn = sys.argv[1]
    xyfn = sys.argv[2]
    
    print('FITS Image', imgfn)
    print('xylist', xyfn)

    p = pyfits.open(imgfn)
    I = p[0].data
    print('Image is', I.shape)

    p = pyfits.open(xyfn)
    xy = p[1].data
    x = xy.field('X')
    y = xy.field('Y')
    flux = xy.field('FLUX')
    print('Sources:', len(x))

    hdr = p[1].header
    sigma = hdr['ESTSIGMA']
    print('Estimated sigma', sigma)

    N = 100

    dx = []
    dy = []
    dx2 = []
    dy2 = []
    fluxes = []
    fluxes2 = []
    Nout = 0
    Nout2 = 0
    for i in range(len(x)):
        #print 'peak',i
        ix = round(x[i]) - 1
        iy = round(y[i]) - 1
        cutout = I[iy-1:iy+2, ix-1:ix+2]
        if cutout.shape != (3,3):
            print('cutout is', cutout.shape)
            continue
        #cutout5 = I[range(iy-2, iy+3, 2), range(ix-2, ix+3, 2)]
        cutout5 = array([ I[range(iy-2, iy+3, 2), ix-2],
                          I[range(iy-2, iy+3, 2), ix  ],
                          I[range(iy-2, iy+3, 2), ix+2], ])
                          
        if cutout5.shape != (3,3):
            print('cutout5 has shape', cutout5.shape)
            print('yrange', range(iy-2, iy+3, 2))
            print('xrange', range(ix-2, ix+3, 2))
            print('cutout5', cutout5)
            continue

        for j in xrange(N):
            noise = normal(0, sigma, size=(3,3))
            img = cutout + noise
            cen = dcen3x3(img)
            # original center:
            xx = x[i] - ix
            yy = y[i] - iy
            if cen is not None:
                (cx,cy) = cen
                dx.append(cx - xx)
                dy.append(cy - yy)
                fluxes.append(flux[i])
            else:
                Nout += 1
                noise5 = normal(0, sigma, size=(3,3))
                cen2 = dcen3x3(cutout5 + noise5)
                if cen2 is None:
                    Nout2 += 1
                    continue
                (cx2, cy2) = cen2
                cx = 1. + (cx2-1.) * 2.
                cy = 1. + (cy2-1.) * 2.
                dx2.append(cx - xx)
                dy2.append(cy - yy)
                fluxes2.append(flux[i])


    dx = array(dx)
    dy = array(dy)
    dx2 = array(dx2)
    dy2 = array(dy2)

    print('A total of', Nout,  'of', (N*len(x)), '(%i %%)' % int(round(Nout*100./float(N*len(x)))),  'peaks moved outside the 3x3 box.')
    print('A total of', Nout2, 'of', (N*len(x)), '(%i %%)' % int(round(Nout2*100./float(N*len(x)))), 'peaks moved outside the 5x5 box.')
    figure()
    clf()
    subplot(1,2,1)
    plot(dx, dy, 'r.')
    axis('equal')
    ylim(-3,3)
    a=axis()
    axhline(0)
    axvline(0)
    axis(a)
    title('3x3 centroid error')
    xlabel('distance (pixels)')
        
    subplot(1,2,2)
    plot(dx2, dy2, 'b.')
    axis('equal')
    ylim(-3,3)
    axhline(0)
    axvline(0)
    axis(a)
    title('5x5 centroid error')
    xlabel('distance (pixels)')
    savefig('dxdy.png', dpi=75)
    subplot(111)

    dist = sqrt(dx**2 + dy**2)
    dist2 = sqrt(dx2**2 + dy2**2)
    clf()
    subplot(2,1,1)
    (nb,bins,patches) = hist(dist, 40)
    xlim(0,bins[-1])
    title('3x3 centroid error distance')
    #xlabel('pixels')
    subplot(2,1,2)
    hist(dist2, bins=bins)
    xlim(0,bins[-1])
    title('5x5 centroid error distance')
    xlabel('pixels')
    savefig('dists.png', dpi=75)

    clf()

    dboth = hstack((dx,dy))
    sigd = std(dboth)
    md = mean(dboth)
    xx = arange(dboth.min(), dboth.max(), 0.01)
    yy = (len(dboth)*(xx.max()-xx.min())/40) / (sigd * sqrt(2. * pi)) * exp(-((xx-md)**2)/(2*sigd**2))

    subplot(2,1,1)
    (n,bins,patches) = hist(dboth, 40)
    plot(xx, yy, 'r-')
    title('3x3 centroid coordinate errors (std=%g)' % sigd)
    #xlabel('pixels')

    dboth2 = hstack((dx2,dy2))
    sigd2 = std(dboth2)
    md2 = mean(dboth2)
    if len(dboth2):
        xx2 = arange(dboth2.min(), dboth2.max(), 0.01)
        yy2 = (len(dboth2)*(xx2.max()-xx2.min())/40) / (sigd2 * sqrt(2. * pi)) * exp(-((xx2-md2)**2)/(2*sigd2**2))
    else:
        xx2 = []
        yy2 = []
    subplot(2,1,2)
    hist(dboth2, bins=bins)
    plot(xx2, yy2, 'r-')
    title('5x5 centroid coordinate errors (std=%g)' % sigd2)
    xlabel('pixels')
    savefig('dboth.png', dpi=75)

    clf()
    subplot(2,1,1)
    plot(fluxes, dist, 'r.')
    ylabel('pixel distance')
    #xlabel('flux')
    title('3x3 centroid')
    subplot(2,1,2)
    plot(fluxes2, dist2, 'b.')
    ylabel('pixel distance')
    xlabel('flux')
    title('5x5 centroid')
    savefig('fluxdist.png', dpi=75)
    
    

