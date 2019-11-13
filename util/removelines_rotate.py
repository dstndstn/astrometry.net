#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import
import matplotlib
matplotlib.use('Agg')
import os
import sys
import logging

from pylab import *
from optparse import OptionParser

from astrometry.util.removelines import hist_remove_lines
from astrometry.util.fits import pyfits_writeto

if __name__ == '__main__':
    try:
        try:
            import pyfits
        except ImportError:
            try:
                from astropy.io import fits as pyfits
            except ImportError:
                raise ImportError(
                    "Cannot import either pyfits or astropy.io.fits"
                )
        import astrometry
        from astrometry.util.shell import shell_escape
        from astrometry.util.filetype import filetype_short
    except ImportError:
        me = sys.argv[0]
        #print 'i am', me
        path = os.path.realpath(me)
        #print 'my real path is', path
        utildir = os.path.dirname(path)
        assert(os.path.basename(utildir) == 'util')
        andir = os.path.dirname(utildir)
        #assert(os.path.basename(andir) == 'astrometry')
        rootdir = os.path.dirname(andir)
        #print 'adding path', rootdir
        #sys.path += [rootdir]
        sys.path.insert(1, andir)
        sys.path.insert(2, rootdir)

import numpy
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
            raise ImportError("Cannot import either pyfits or astropy.io.fits")
from numpy import *
from numpy.random import rand

'''
# Returns a numpy array of booleans: True for points that should be kept (are not part of lines)
if False:
    # def hist_remove_lines(x, binwidth, binoffset, nsig, plots=False):
    bins = -binoffset + arange(0, max(x)+binwidth, binwidth)
    (counts, thebins) = histogram(x, bins)

    # We're ignoring empty bins.
    occupied = nonzero(counts)[0]
    noccupied = len(occupied)
    #k = (counts[occupied] - 1)
    #mean = sum(k) / float(noccupied)
    k = counts[occupied]
    mean = sum(k) / ((max(x) - min(x)) / binwidth) * sqrt(2.)
    thresh = mean + nsig * sqrt(mean)

    if plots:
        hist(x, bins)
        axhline(mean)
        axhline(thresh, color='r')

    print 'mean', mean, 'thresh:', thresh, 'max:', max(k)

    #logpoisson = k*log(mean) - mean - array([sum(log(1 + arange(kk))) for kk in k])
    #uk = unique(k)
    #ulogpoisson = uk*log(mean) - mean - array([sum(1+arange(kk)) for kk in uk])
    #print
    #for (uuk,ull) in zip(uk,ulogpoisson):
    #    print uuk,ull

    #badbins = occupied[logpoisson < logcut]
    badbins = occupied[k > thresh]
    if len(badbins) == 0:
        return array([True] * len(x))

    badleft = bins[badbins]
    badright = badleft + binwidth

    badpoints = sum(array([(x >= L)*(x < R) for (L,R) in zip(badleft, badright)]), 0)
    return (badpoints == 0)
'''
#'''

def removelines(infile, outfile, xcol='X', ycol='Y', plots=False, cut=None, **kwargs):
    p = pyfits.open(infile)
    xy = p[1].data
    hdr = p[1].header
    x = xy.field(xcol)
    y = xy.field(ycol)

    NX = max(x) - min(x)
    NY = max(y) - min(y)
    nangle = int(ceil(sqrt(NX*NY)/4.))

    if cut is None:
        cut = 20

    if plots:
        clf()
        plot(x, y, 'r.')

    I = array([True]*len(x))
    for i,angle in enumerate(0.75 + linspace(0, pi/2., nangle, endpoint=False)):
        cost = cos(angle)
        sint = sin(angle)
        xx = x*cost     + y*sint
        yy = x*-sint + y*cost
        xx -= min(xx)
        yy -= min(yy)

        if plots:
            print()
            clf()
            subplot(2,2,1)
            plot(xx, yy, 'r.')
            subplot(2,2,3)


        #ix = hist_remove_lines(xx, 0.5, 0.5, 5, plots=plots)
        ix = hist_remove_lines(xx, 1, 0.5, logcut=-cut)

        if plots:
            subplot(2,2,4)

        #iy = hist_remove_lines(yy, 0.5, 0.5, 5, plots=plots)
        iy = hist_remove_lines(yy, 1, 0.5, logcut=-cut)

        I *= ix * iy

        removed = (ix * iy == False)
        if plots:
            if sum(removed):
                plot([min(x[removed]), max(x[removed])],
                     [min(y[removed]), max(y[removed])], 'k-', alpha=0.5)
            subplot(2,2,1)
            plot(xx[removed], yy[removed], 'b-', alpha=0.5)
            plot(xx[removed], yy[removed], 'b.')
            savefig('rot-%04i.png' % i)

        print('angle', angle, 'removed', (len(x) - sum(ix*iy)))

    xc = x[I]
    yc = y[I]
    print('removelines.py: Removed %i sources' % (len(x) - len(xc)))

    if plots:
        plot(xc, yc, 'o', mec='r', mfc='none')
        # axes('equal')
        savefig('after.png')

    p[1].header.add_history('This xylist was filtered by the "removelines_rotate.py" program')
    p[1].header.add_history('to remove lines of sources')
    p[1].header.update('REMLINEN', len(x) - len(xc), 'Number of sources removed by "removelines"')

    p[1].data = p[1].data[I]
    pyfits_writeto(p, outfile)

    return 0

if __name__ == '__main__':
    parser = OptionParser(usage='%prog <input-file> <output-file>')
    parser.add_option('-X', '--x-column', dest='xcol', help='X column name')
    parser.add_option('-Y', '--y-column', dest='ycol', help='Y column name')
    parser.set_defaults(xcol='X', ycol='Y')
    opt,args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(0)

    kwargs = {}
    for k in 'xcol','ycol':
        kwargs[k] = getattr(opt, k)

    infile = args[0]
    outfile = args[1]
    rtncode = removelines(infile, outfile, **kwargs)
    sys.exit(rtncode)

