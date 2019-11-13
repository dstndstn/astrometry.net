#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import pylab as plt
import numpy as np
import sys
from math import sqrt, floor, ceil
from astrometry.util.util import Sip
from optparse import OptionParser

def plotDistortion(sip, W, H, ncells, exaggerate=1., doclf=True):
    '''
    Produces a plot showing the SIP distortion that was found, by drawing
    a grid and distorting it.  Allows exaggeration of the distortion for ease
    of visualization.

    sip -- an astrometry.util.Sip object
       (duck-type: has "(dx,dy) = sip.get_distortion(x,y)")
    W, H -- the image size
    ncells -- the approximate number of grid cells to split the image into.
    exaggerate -- the factor by which to exaggerate the distortion.
    
    '''
    ncells = float(ncells)
    cellsize = sqrt(W * H / ncells)
    nw = int(floor(W / cellsize))
    nh = int(floor(H / cellsize))
    #print 'Grid cell size', cellsize
    #print 'N cells', nw, 'x', nh
    cx = np.arange(nw+1) * cellsize + ((W - (nw*cellsize))/2.)
    cy = np.arange(nh+1) * cellsize + ((H - (nh*cellsize))/2.)

    # pixel step size for grid lines
    step = 50

    nx = int(np.ceil(W / float(step)))
    ny = int(np.ceil(H / float(step)))
    
    #xx = np.arange(-step, W+2*step, step)
    #yy = np.arange(-step, H+2*step, step)
    xx = np.linspace(1, W, nx)
    yy = np.linspace(1, H, ny)
    
    if doclf:
        plt.clf()

    for y in cy:
        dx,dy = [],[]
        for x in xx:
            dxi,dyi = sip.get_distortion(x, y)
            dx.append(dxi)
            dy.append(dyi)
        plt.plot(xx, y*np.ones_like(xx), 'k-', zorder=10)
        dx = np.array(dx)
        dy = np.array(dy)
        if exaggerate != 1:
            dx += (exaggerate * (dx - xx))
            dy += (exaggerate * (dy - y))
        plt.plot(dx, dy, 'r-', zorder=20)

    for x in cx:
        dx,dy = [],[]
        for y in yy:
            dxi,dyi = sip.get_distortion(x, y)
            dx.append(dxi)
            dy.append(dyi)
        plt.plot(x*np.ones_like(yy), yy, 'k-', zorder=10)
        dx = np.array(dx)
        dy = np.array(dy)
        if exaggerate != 1:
            dx += (exaggerate * (dx - x))
            dy += (exaggerate * (dy - yy))
        plt.plot(dx, dy, 'r-', zorder=20, clip_on=False)
    
    plt.axis('scaled')
    plt.axis([0, W, 0, H])

def plotDistortionFile(sipfn, ext, ncells, **kwargs):
    wcs = Sip(sipfn, ext)
    if wcs is None:
        raise RuntimeError('Failed to open WCS file %s' % sipfn)
    plotDistortion(wcs, wcs.get_width(), wcs.get_height(), ncells, **kwargs)



if __name__ == '__main__':
    parser = OptionParser('usage: %prog [options] <wcs-filename> <plot-output-filename>')
    parser.add_option('-e', dest='ext', type='int', help='FITS extension to read WCS from (default 0)')
    parser.add_option('-x', dest='exaggerate', type='float', help='Exaggeration factor')
    parser.add_option('-c', dest='cells', type='int', help='Approx. number of pieces to cut image into (default:18)')
    parser.set_defaults(ext=0, cells=18, exaggerate=1.)
    opt,args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)

    wcsfn = args[0]
    outfn = args[1]

    plotDistortionFile(wcsfn, opt.ext, opt.cells, exaggerate=opt.exaggerate)
    plt.savefig(outfn)
