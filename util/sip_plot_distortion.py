# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import sys
from optparse import *

import numpy as np
from pylab import *
from numpy import *
#from astrometry.util.sip import *
from astrometry.util.util import *

def plot_distortions(wcsfn, ex=1, ngridx=10, ngridy=10, stepx=10, stepy=10):
    wcs = Sip(wcsfn)
    W,H = wcs.wcstan.imagew, wcs.wcstan.imageh

    xgrid = np.linspace(0, W, ngridx)
    ygrid = np.linspace(0, H, ngridy)
    X = np.linspace(0, W, int(ceil(W/stepx)))
    Y = np.linspace(0, H, int(ceil(H/stepy)))

    xlo,xhi,ylo,yhi = 0,W,0,H

    for x in xgrid:
        DX,DY = [],[]
        xx,yy = [],[]
        UX,UY = [],[]
        for y in Y:
            dx,dy = wcs.get_distortion(x, y)
            ux,uy = wcs.get_undistortion(dx, dy)
            print('x,y', (x,y), 'dx,dy', (dx,dy), 'ux,uy', (ux,uy))
            xx.append(x)
            yy.append(y)
            DX.append(dx)
            DY.append(dy)
            UX.append(ux)
            UY.append(uy)
        DX = np.array(DX)
        DY = np.array(DY)
        UX = np.array(UX)
        UY = np.array(UY)
        xx = np.array(xx)
        yy = np.array(yy)
        EX = DX + ex * (DX - xx)
        EY = DY + ex * (DY - yy)
        plot(xx, yy, 'k-', alpha=0.5)
        plot(EX, EY, 'r-')
        plot(UX, UY, 'b-', alpha=0.5)
        xlo = min(xlo, min(EX))
        xhi = max(xhi, max(EX))
        ylo = min(ylo, min(EY))
        yhi = max(yhi, max(EY))

    for y in ygrid:
        DX,DY = [],[]
        xx,yy = [],[]
        UX,UY = [],[]
        for x in X:
            dx,dy = wcs.get_distortion(x, y)
            ux,uy = wcs.get_undistortion(dx, dy)
            DX.append(dx)
            DY.append(dy)
            UX.append(ux)
            UY.append(uy)
            xx.append(x)
            yy.append(y)
        DX = np.array(DX)
        DY = np.array(DY)
        xx = np.array(xx)
        yy = np.array(yy)
        EX = DX + ex * (DX - xx)
        EY = DY + ex * (DY - yy)
        plot(xx, yy, 'k-', alpha=0.5)
        plot(EX, EY, 'r-')
        plot(UX, UY, 'b-', alpha=0.5)
        xlo = min(xlo, min(EX))
        xhi = max(xhi, max(EX))
        ylo = min(ylo, min(EY))
        yhi = max(yhi, max(EY))

    plot([wcs.wcstan.crpix[0]], [wcs.wcstan.crpix[1]], 'rx')

    #axis([0, W, 0, H])
    axis('scaled')
    axis([xlo,xhi,ylo,yhi])
    #axis('tight')

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <wcs-filename> <plot-filename>')
    parser.add_option('-e', '--ex', '--exaggerate', dest='ex', type='float', help='Exaggerate the distortion by this factor')
    #parser.add_option('-s', '--scale', dest='scale', type='float', help='Scale the
    parser.add_option('-n', dest='nsteps', type='int', help='Number of grid lines to plot')
 
    parser.set_defaults(ex=1.)

    opt,args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)

    wcsfn = args[0]
    outfn = args[1]

    args = {}
    if opt.ex is not None:
        args['ex'] = opt.ex
    if opt.nsteps is not None:
        args['ngridx'] = opt.nsteps
        args['ngridy'] = opt.nsteps

    clf()
    plot_distortions(wcsfn, **args)
    tt = 'SIP distortions: %s' % wcsfn
    if opt.ex != 1:
        tt += ' (exaggerated by %g)' % opt.ex
    title(tt)
    savefig(outfn)
    
