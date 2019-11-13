#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function

import sys
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")

from math import *
from numpy import *
from pylab import *
from scipy.ndimage.filters import *
from astrometry.util.fits import pyfits_writeto

def normalized_hough(x, y, imgw, imgh, rlo, rhi, tlo, thi, nr, nt):
    houghimg = zeros((nr, nt)).astype(int)
    tstep = (thi - tlo) / float(nt)
    rstep = (rhi - rlo) / float(nr)
    # For each point, accumulate into the Hough transform image...
    tt = tlo + (arange(nt) + 0.5) * tstep
    cost = cos(tt)
    sint = sin(tt)
    for (xi,yi) in zip(x, y):
        rr = xi * cost + yi * sint
        ri = floor((rr - rlo) / rstep).astype(int)
        I = (ri >= 0) * (ri < nr)
        houghimg[ri[I], I] += 1

    # Compute the approximate Hough normalization image
    houghnorm = zeros((nr, nt)).astype(float)
    rr = rlo + (arange(nr) + 0.5) * rstep
    for ti,t in enumerate(tt):
        (x0,x1,y0,y1) = clip_to_image(rr, t, imgw, imgh)
        dist = sqrt((x0 - x1)**2 + (y0 - y1)**2)
        houghnorm[:, ti] = dist
    # expected number of points: dist is the length of the slice,
    #   rstep is the width of the slice; len(x)/A is the source density.
    houghnorm *= (rstep * len(x) / (imgw*imgh))
    return (houghimg, houghnorm, rr, tt, rstep, tstep)

def clip_to_image(r, t, imgw, imgh):
    eps = 1e-9
    if abs(t) < eps or abs(t-pi) < eps:
        # near-vertical.
        s = (abs(t) < eps) and 1 or -1
        y0 = 0
        y1 = ((r*s >= 0) * (r*s < imgw)) * imgh
        x0 = x1 = clip(r, 0, imgw)
        return (x0, x1, y0, y1)
    m = -cos(t)/sin(t)
    b = r/sin(t)
    x0 = 0
    x1 = imgw
    y0 = clip(b + m*x0, 0, imgh)
    y1 = clip(b + m*x1, 0, imgh)
    x0 = clip((y0 - b) / m, 0, imgw)
    x1 = clip((y1 - b) / m, 0, imgw)
    y0 = clip(b + m*x0, 0, imgh)
    y1 = clip(b + m*x1, 0, imgh)
    return (x0, x1, y0, y1)

def removelines_general(infile, outfile, nt=180, nr=180, thresh1=2.,
                        thresh2=5., plots=False):
    p = pyfits.open(infile)
    xy = p[1].data
    hdr = p[1].header
    x = xy.field('X').copy()
    y = xy.field('Y').copy()

    imshowargs = { 'interpolation':'nearest', 'origin':'lower' }

    imgw = int(ceil(max(x) - min(x)))
    imgh = int(ceil(max(y) - min(y)))

    x -= min(x)
    y -= min(y)

    Rmax = sqrt(imgw**2 + imgh**2)
    Rmin = -Rmax

    (houghimg, houghnorm, rr, tt, rstep, tstep
     ) = normalized_hough(x, y, imgw, imgh, Rmin, Rmax, 0, pi, nr, nt)

    hnorm = houghimg / maximum(houghnorm, 1)

    if plots:
        clf()
        plot(x,y,'r.')
        savefig('xy.png')

        clf()
        imshow(houghimg, **imshowargs)
        xlabel('Theta')
        ylabel('Radius')
        colorbar()
        savefig('hough.png')

        clf()
        imshow(houghnorm, **imshowargs)
        xlabel('Theta')
        ylabel('Radius')
        colorbar()
        savefig('norm.png')

        clf()
        imshow(hnorm, **imshowargs)
        xlabel('Theta')
        ylabel('Radius')
        colorbar()
        savefig('hnorm.png')


    I = find(hnorm.ravel() >= thresh1)
    print('%i peaks are above the coarse threshold' % len(I))
    bestri = I / nt
    bestti = I % nt

    if plots:
        a=axis()
        for (ri,ti) in zip(bestri,bestti):
            plot([ti-2, ti-2, ti+2, ti+2, ti-2], [ri-2, ri+2, ri+2, ri-2, ri-2], 'r-')
            axis(a)
            savefig('zooms.png')

        clf()
        plot(x,y,'r.')
        for (ri,ti) in zip(bestri,bestti):
            (x0,x1,y0,y1) = clip_to_image(rr[ri], tt[ti], imgw, imgh)
            plot([x0,x1],[y0,y1], 'b-')
        savefig('xy2.png')

    # how big a search area around each peak?
    boxsize = 1
    # how much more finely to grid.
    finer = 3
    nr2 = (boxsize * 2)*finer + 1
    nt2 = nr2

    bestrt = []
    keep = array([True] * len(x))
    for (ri,ti) in zip(bestri,bestti):
        (subh, subnorm, subrr, subtt, subrstep, subtstep
         ) = normalized_hough(x, y, imgw, imgh,
                              rr[max(ri-boxsize, 0)], rr[min(ri+boxsize, nr-1)],
                              tt[max(ti-boxsize, 0)], tt[min(ti+boxsize, nt-1)],
                              nr2, nt2)
        #print '  median normalization:', median(subnorm)
        subhnorm = subh / maximum(subnorm,1)
        I = find((subhnorm).ravel() >= thresh2)
        for i in I:
            bestsubri = i / nt2
            bestsubti = i % nt2
            r = subrr[bestsubri]
            t = subtt[bestsubti]
            bestrt.append((r,t))
            #print '  (r=%.1f, t=%.1f): factor %.1f above expected' % (r, t*180/pi, subhnorm.ravel()[i])
            thisr = x * cos(t) + y * sin(t)
            keep *= (abs(thisr - r) > subrstep/2.)

    print('In finer grid: found %i peaks' % len(bestrt))

    if plots:
        clf()
        subplot(1,1,1)
        plot(x,y,'r.')
        for (r,t) in bestrt:
            (x0,x1,y0,y1) =  clip_to_image(r, t, imgw, imgh)
            plot([x0,x1],[y0,y1],'b-')
        savefig('xy3.png')

        clf()
        plot(x,y,'r.')
        plot(x[keep == False], y[keep == False], 'b.')
        savefig('xy4.png')

    p[1].data = p[1].data[keep]
    pyfits_writeto(p, outfile)
    return 0

def exact_hough_normalization():
    houghnorm = zeros((nr, nt)).astype(float)
    [xx,yy] = meshgrid(arange(imgw), arange(imgh))
    yyflat = yy.ravel()
    xxflat = xx.ravel()
    for ti in range(nt):
        print(ti)
        t = (ti+0.5) * tstep
        rr = xxflat * cos(t) + yyflat * sin(t)
        ri = floor((rr - Rmin) / rstep).astype(int)
        (counts, nil) = histogram(ri, range(0, nr+1))
        houghnorm[:, ti] += counts
    clf()
    imshow(houghnorm, **imshowargs)
    colorbar()
    savefig('houghnorm.png')
    


if __name__ == '__main__':
    args = sys.argv[1:]
    plots = False
    if '-p' in args:
        plots = True
        args.remove('-p')

    if len(args) != 2:
        print('Usage: %s [options] <input-file> <output-file>' % sys.argv[0])
        print('   [-p]: create plots')
        exit(-1)

    infile = args[0]
    outfile = args[1]
    rtncode = removelines_general(infile, outfile, plots=plots)
    sys.exit(rtncode)

