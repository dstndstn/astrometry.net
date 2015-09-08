# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import math
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
from numpy import *
from matplotlib.pylab import figure, plot, xlabel, ylabel, loglog, clf
from matplotlib.pylab import semilogy, show, find, legend, hist, axis

def plotDvsR(ix, iy, fx, fy, R):
    IR = argsort(R)

    mR = []
    mD = []
    mI = []

    # Look at index stars in order of R.
    for ii in range(len(IR)):
        i = IR[ii]
        x = ix[i]
        y = iy[i]
        D = sqrt((fx - x)**2 + (fy - y)**2)
        # Grab field stars within a matching radius.
        r = 5
        I = find( D < r )
        for j in I:
            mR.append(R[i])
            mD.append(D[j])
            mI.append(ii)
    plot(mR, mD, 'ro')
    xlabel('Distance from quad center')
    ylabel('Match distance')

def plotIvsR(ix, iy, fx, fy, cx, cy):
    RF = sqrt((fx - cx)**2 + (fy - cy)**2)
    mD2 = []
    mI2 = []
    for i in range(len(RF)):
        r = 5
        D = sqrt((ix - fx[i])**2 + (iy - fy[i])**2)
        I = find( D < r )
        for j in I:
            mD2.append(D[j])
            mI2.append(i)
    #plot(mI, mD, 'ro', mI2, mD2, 'bo')
    plot(mI2, mD2, 'bo')
    xlabel('Index star number')
    ylabel('Match distance')
    #legend(('Sorted by dist from quad center', 'Sorted by brightness'))


    

if __name__ == '__main__':

    # Index stars
    ixy = pyfits.open('ver/index.xy.fits')
    ixy = ixy[1].data
    ix = ixy.field(0)
    iy = ixy.field(1)
    NI = len(ix)

    # Field stars
    fxy = pyfits.open('ver/field.xy.fits')
    fxy = fxy[1].data
    fx = fxy.field(0)
    fy = fxy.field(1)
    NF = len(fx)

    # The matched quad.
    mf = pyfits.open('ver/match.fits')
    mf = mf[1].data
    quad = mf.field('quadpix')[0]
    quad = quad[0:8].reshape(4,2)
    qx = quad[:,0]
    qy = quad[:,1]
    # Quad center.
    cx = mean(qx)
    cy = mean(qy)

    # Grab index stars that are within the field.
    iok = find( (ix > min(fx)) *
                (ix < max(fx)) *
                (iy > min(fy)) *
                (iy < max(fy)) )
    ix = [ix[i] for i in iok]
    iy = [iy[i] for i in iok]


    figure(1)
    clf()
    I = [0, 2, 1, 3, 0]
    plot(
        [cx], [cy], 'ro',
        qx[I], qy[I], 'r-',
        ix, iy, 'rx',
        fx, fy, 'b+',
        )

    # RMS quad radius
    RQ = sqrt(sum((qx - cx)**2 + (qy - cy)**2) / 4)

    # Distance from quad center.
    RI = sqrt((ix - cx)**2 + (iy - cy)**2)
    RF = sqrt((fx - cx)**2 + (fy - cy)**2)

    # Angle from quad center.
    AI = array([math.atan2(y - cy, x - cx) for (x,y) in zip(ix,iy)])
    AF = array([math.atan2(y - cy, x - cx) for (x,y) in zip(fx,fy)])

    # Look at index stars in order of R.
    IR = argsort(RI)
    allD = array([])
    allDR = array([])
    allDA = array([])
    allR = array([])
    allDist = array([])

    for i in IR:
        # regularizer...
        reg = RQ
        #
        Distscale = 1
        DRscale = 1 / RQ

        dR = ((RI[i] + reg) / (RF + reg)) - 1.0
        dA =  AI[i] - AF
        # handle wrap-around
        absdA = abs(dA)
        absdA = vstack((absdA, abs(absdA - 2*math.pi))).min(axis=0)
        #D = sqrt(dR**2 + dA**2)
        D = sqrt(dR**2 + absdA**2)
        D = D / DRscale

        Dist = sqrt((ix[i] - fx)**2 + (iy[i] - fy)**2)
        Dist = Dist / Distscale

        iSmall = array(find((D < 1) + (Dist < 1)))
        allD = hstack((allD, D[iSmall]))
        allDR = hstack((allDR, dR[iSmall]))
        allDA = hstack((allDA, dA[iSmall]))
        allR = hstack((allR, repeat(RI[i], len(iSmall))))
        allDist = hstack((allDist, Dist[iSmall]))

    figure(2)
    clf()
    plot(allDR, allDA, 'ro', ms=1)
    xlabel('DR')
    ylabel('DA')

    figure(3)
    clf()
    #plot(allR/RQ, allD, 'r.')
    #plot(allR/RQ, vstack((allD, allDist)).min(axis=0), 'r.')
    plot(
        #allR/RQ, vstack((allD, allDist)).min(axis=0), 'mo',
        allR/RQ, allD, 'r+',
        allR/RQ, allDist, 'bx'
        )
    xlabel('R (quad radiuses)')
    #ylabel('min( Dist, D(R+A) )')
    ylabel('Dist, D(R,A)')
    a = axis()
    axis([a[0], a[1], 0, 2.0])
    legend(('D(R,A)', 'D'))

    #allDist = array([])
    #allRDist = array([])
    #for i in IR:
    #    Dist = sqrt((ix[i] - fx)**2 + (iy[i] - fy)**2)
    #    iSmall = array(find(Dist < 5))
    #    allDist = hstack((allDist, Dist[iSmall]))
    #    allRDist = hstack((allRDist, repeat(RI[i], len(iSmall))))
    #figure(4)
    #clf()
    #plot(allRDist/RQ, allDist, 'r.')
    #xlabel('R')
    #ylabel('Dist')

    #figure(2)
    #clf()
    #plotDvsR(ix, iy, RI, fx, fy)

    #figure(3)
    #clf()
    #plotDvsI(ix, iy, fx, fy, cx, cy)

    
