#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import logging
from optparse import OptionParser
import numpy as np
from astrometry.util.fits import fits_table

# Returns a numpy array of booleans
def hist_remove_lines(x, binwidth, binoffset, logcut):
    bins = -binoffset + np.arange(0, max(x)+binwidth+1, binwidth)
    (counts, thebins) = np.histogram(x, bins)

    # We're ignoring empty bins.
    occupied = np.nonzero(counts > 1)[0]
    noccupied = len(occupied)
    if noccupied == 0:
        return np.array([True] * len(x))
    k = (counts[occupied] - 1) 
    mean = sum(k) / float(noccupied)
    logpoisson = k*np.log(mean) - mean - np.array([sum(np.arange(kk)) for kk in k])
    badbins = occupied[logpoisson < logcut]
    if len(badbins) == 0:
        return np.array([True] * len(x))

    badleft = bins[badbins]
    badright = badleft + binwidth

    badpoints = sum(np.array([(x >= L)*(x < R) for (L,R) in
                              zip(badleft, badright)]), 0)
    return (badpoints == 0)

def removelines(infile, outfile, xcol='X', ycol='Y', ext=1, cut=None, **kwargs):
    if cut is None:
        cut = 100
    T = fits_table(infile, lower=False)
    if len(T) == 0:
        print('removelines.py: Input file contains no sources.')
        T.writeto(outfile)
        return 0
    
    ix = hist_remove_lines(T.get(xcol), 1, 0.5, logcut=-cut)
    iy = hist_remove_lines(T.get(ycol), 1, 0.5, logcut=-cut)
    Norig = len(T)
    T.cut(ix * iy)
    print('removelines.py: Removed %i sources' % (Norig - len(T)))
    #header.add_history('This xylist was filtered by the "removelines.py" program')
    #header.add_history('to remove horizontal and vertical lines of sources')
    #header['REMLINEN'] = (len(x) - len(xc), 'Number of sources removed by "removelines.py"')
    T.writeto(outfile, header=T.get_header())
    return 0

def main():
    parser = OptionParser(usage='%prog [options] <input-xylist> <output-xylist>')

    parser.add_option('-X', dest='xcol', help='Name of X column in input table')
    parser.add_option('-Y', dest='ycol', help='Name of Y column in input table')
    parser.add_option('-s', dest='cut', type='float', help='Significance level to cut at (default 100)')
    parser.add_option('-e', dest='ext', type=int, help='FITS extension to read',
                      default=1)
    parser.set_defaults(xcol='X', ycol='Y', cut=None)

    (opt, args) = parser.parse_args()
    
    if len(args) != 2:
        parser.print_help()
        print()
        print('Got arguments:', args)
        sys.exit(-1)

    infile = args[0]
    outfile = args[1]
    return removelines(infile, outfile, xcol=opt.xcol, ycol=opt.ycol,
                       cut=opt.cut, ext=opt.ext)

if __name__ == '__main__':
    sys.exit(main())
