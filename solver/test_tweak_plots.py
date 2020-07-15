# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os.path
from pylab import *
from numpy import *
# test_tweak 2>tt.py
import tt

def savefig(fn):
    from pylab import savefig as sf
    print('Saving', fn)
    sf(fn)

if __name__ == '__main__':
    #print 'me:', __file__
    #tt = os.path.join(os.path.dirname(__file__), 'test_tweak')

    for run in [2,3]:

        tanxy = getattr(tt, 'origxy_%i' % run)
        xy = getattr(tt, 'xy_%i' % run)
        noisyxy = getattr(tt, 'noisyxy_%i' % run)
        gridx = getattr(tt, 'gridx_%i' % run)
        gridy = getattr(tt, 'gridy_%i' % run)
        truesip_a = getattr(tt, 'truesip_a_%i' % run)
        truesip_b = getattr(tt, 'truesip_b_%i' % run)
        sip_a = getattr(tt, 'sip_a_%i' % run)
        sip_b = getattr(tt, 'sip_b_%i' % run)
        x0,y0 = tt.x0, tt.y0
        
        truedxy = xy - tanxy
        obsdxy  = noisyxy - tanxy

        xlo,xhi = -500, 2500
        ylo,yhi = -500, 2500

        X1 = linspace(xlo, xhi, 100)
        Y1 = gridy
        X1,Y1 = meshgrid(X1,Y1)
        X1 = X1.T
        Y1 = Y1.T
        X2 = gridx
        Y2 = linspace(ylo, yhi, 100)
        X2,Y2 = meshgrid(X2,Y2)
        truesipx_x = zeros_like(X1)
        truesipy_x = zeros_like(X1)
        truesipx_y = zeros_like(Y2)
        truesipy_y = zeros_like(Y2)
        for xo,yo,c in truesip_a:
            truesipx_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
            truesipx_x += c * (X1 - x0)**xo * (Y1 - y0)**yo
        for xo,yo,c in truesip_b:
            truesipy_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
            truesipy_x += c * (X1 - x0)**xo * (Y1 - y0)**yo

        x = xy[:,0]
        y = xy[:,1]
        truedx = truedxy[:,0]
        truedy = truedxy[:,1]
        obsdx = obsdxy[:,0]
        obsdy = obsdxy[:,1]

        for order in range(2,6):
            clf()

            sipx_x = zeros_like(X1)
            sipy_x = zeros_like(X1)
            sipx_y = zeros_like(Y2)
            sipy_y = zeros_like(Y2)
            for xo,yo,c in sip_a[order]:
                sipx_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
                sipx_x += c * (X1 - x0)**xo * (Y1 - y0)**yo
            for xo,yo,c in sip_b[order]:
                sipy_y += c * (X2 - x0)**xo * (Y2 - y0)**yo
                sipy_x += c * (X1 - x0)**xo * (Y1 - y0)**yo

            subplot(2,2,1)
            plot(x, truedx, 'bs', mec='b', mfc='None')
            plot(x, obsdx, 'r.')
            plot(X1, -truesipx_x, 'b-', alpha=0.2)
            plot(X1, -sipx_x, 'r-', alpha=0.2)
            xlabel('x')
            ylabel('dx')
            xlim(xlo, xhi)

            subplot(2,2,2)
            plot(x, truedy, 'bs', mec='b', mfc='None')
            plot(x, obsdy, 'r.')
            plot(X1, -truesipy_x, 'b-', alpha=0.2)
            plot(X1, -sipy_x, 'r-', alpha=0.2)
            xlabel('x')
            ylabel('dy')
            xlim(xlo, xhi)

            subplot(2,2,3)
            plot(y, truedx, 'bs', mec='b', mfc='None')
            plot(y, obsdx, 'r.')
            plot(Y2, -truesipx_y, 'b-', alpha=0.2)
            plot(Y2, -sipx_y, 'r-', alpha=0.2)
            xlabel('y')
            ylabel('dx')
            xlim(xlo, xhi)

            subplot(2,2,4)
            plot(y, truedy, 'bs', mec='b', mfc='None')
            plot(y, obsdy, 'r.')
            plot(Y2, -truesipy_y, 'b-', alpha=0.2)
            plot(Y2, -sipy_y, 'r-', alpha=0.2)
            xlabel('y')
            ylabel('dy')
            xlim(xlo, xhi)

            savefig('tt%i-%i.png' % (run, order))
    
            clf()
            subplot(111)

            plot(tanxy[:,0], tanxy[:,1], 'b.')
            plot(noisyxy[:,0], noisyxy[:,1], 'r.')
            plot(xy[:,0], xy[:,1], 'bo', mec='b', mfc='None')

            if False:
                X3,Y3 = meshgrid(linspace(xlo, xhi, 11),
                                 linspace(ylo, yhi, 11))
                truesipx = X3
                truesipy = Y3
                for xo,yo,c in truesip_a:
                    truesipx += c * (X3 - x0)**xo * (Y3 - y0)**yo
                for xo,yo,c in truesip_b:
                    truesipy += c * (X3 - x0)**xo * (Y3 - y0)**yo
                sipx = X3
                sipy = Y3
                for xo,yo,c in sip_a[order]:
                    sipx += c * (X3 - x0)**xo * (Y3 - y0)**yo
                for xo,yo,c in sip_b[order]:
                    sipy += c * (X3 - x0)**xo * (Y3 - y0)**yo
                plot(truesipx, truesipy, 'bs', mec='b', mfc='None')
                plot(sipx, sipy, 'ms', mec='m', mfc='None')

            plot(X1, Y1, 'g-', alpha=0.25)
            plot(X2, Y2, 'g-', alpha=0.25)

            plot(truesipx_x + X1, truesipy_x + Y1, 'b-', alpha=0.25)
            plot(truesipx_y + X2, truesipy_y + Y2, 'b-', alpha=0.25)

            plot(sipx_x + X1, sipy_x + Y1, 'r-', alpha=0.25)
            plot(sipx_y + X2, sipy_y + Y2, 'r-', alpha=0.25)

            xlim(xlo,xhi)
            ylim(ylo,yhi)
            savefig('ttxy%i-%i.png' % (run,order))
