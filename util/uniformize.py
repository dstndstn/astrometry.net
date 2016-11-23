#! /usr/bin/env python
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import os
import sys
import logging
from optparse import OptionParser

import numpy as np
from astrometry.util.fits import pyfits_writeto
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")

def uniformize(infile, outfile, n, xcol='X', ycol='Y', ext=1, **kwargs):
    p = pyfits.open(infile)
    xy = p[ext].data
    if xy is None:
        print('No sources')
        pyfits_writeto(p, outfile)
        return
    hdr = p[ext].header
    x = xy.field(xcol)
    y = xy.field(ycol)
    if len(x) == 0:
        print('Empty xylist')
        pyfits_writeto(p, outfile)
        return
    
    # use IMAGEW,H, or compute bounds?
    #  #$)(*&%^ NaNs in LSST source positions.  Seriously, WTF!
    I = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not all(I):
        print('%i source positions are not finite.' % np.sum(np.logical_not(I)))
        x = x[I]
        y = y[I]
    
    W = max(x) - min(x)
    H = max(y) - min(y)
    if W == 0 or H == 0:
        print('Area of the rectangle enclosing all image sources: %i x %i' % (W,H))
        pyfits_writeto(p, outfile)
        return
    NX = int(max(1, np.round(W / np.sqrt(W*H / float(n)))))
    NY = int(max(1, np.round(n / float(NX))))
    print('Uniformizing into %i x %i bins' % (NX, NY))
    print('Image bounds: x [%g,%g], y [%g,%g]' % (min(x),max(x),min(y),max(y)))

    ix = (np.clip(np.floor((x - min(x)) / float(W) * NX), 0, NX-1)).astype(int)
    iy = (np.clip(np.floor((y - min(y)) / float(H) * NY), 0, NY-1)).astype(int)
    assert(np.all(ix >= 0))
    assert(np.all(ix < NX))
    assert(np.all(iy >= 0))
    assert(np.all(iy < NY))
    I = iy * NX + ix
    assert(np.all(I >= 0))
    assert(np.all(I < NX*NY))
    bins = [[] for i in range(NX*NY)]
    for j,i in enumerate(I):
        bins[int(i)].append(j)
    maxlen = max([len(b) for b in bins])
    J = []
    for i in range(maxlen):
        thisrow = []
        for b in bins:
            if i >= len(b):
                continue
            thisrow.append(b[i])
        thisrow.sort()
        J += thisrow
    J = array(J)
    p[ext].header.add_history('This xylist was filtered by the "uniformize.py" program')
    p[ext].data = p[ext].data[J]
    pyfits_writeto(p, outfile)
    return 0

def main():
    parser = OptionParser(usage='%prog [options] <input-xylist> <output-xylist>')

    parser.add_option('-X', dest='xcol', help='Name of X column in input table',
                      default='X')
    parser.add_option('-Y', dest='ycol', help='Name of Y column in input table',
                      default='Y')
    parser.add_option('-n', dest='n', type='int', default=10,
                      help='Number of boxes, approximately')
    parser.add_option('-e', dest='ext', type=int, help='FITS extension to read',
                      default=1)
    (opt, args) = parser.parse_args()
    
    if len(args) != 2:
        parser.print_help()
        print()
        print('Got arguments:', args)
        sys.exit(-1)

    infile = args[0]
    outfile = args[1]
    return uniformize(infile, outfile, opt.n, xcol=opt.xcol, ycol=opt.ycol,
                      ext=opt.ext)

if __name__ == '__main__':
    sys.exit(main())

