# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.util import *
from astrometry.util.starutil_numpy import *

def plot_healpix_boundaries(nside):
    for hp in range(12*nside**2):
        # walk healpix boundary in ra,dec
        rd = []
        dd = np.linspace(0, 1, 100)
        x = 0
        rd.extend([healpix_to_radecdeg(hp, nside, x, y)
                   for y in dd])
        y = 1
        rd.extend([healpix_to_radecdeg(hp, nside, x, y)
                   for x in dd])
        x = 1
        rd.extend([healpix_to_radecdeg(hp, nside, x, y)
                   for y in reversed(dd)])
        y = 0
        rd.extend([healpix_to_radecdeg(hp, nside, x, y)
                   for x in reversed(dd)])

        rd = np.array(rd)
        ra,dec = rd[:,0], rd[:,1]

        # Put label in center of healpix
        xyz = radectoxyz(ra, dec)
        xyz = np.mean(xyz, axis=0)
        rc,dc = xyztoradec(xyz)
        plt.text(rc, dc, '%i' % hp, color='b')

        # handle RA wrap-around
        if rc > 180:
            ra += (ra < 90)*360
        else:
            ra -= (ra > 270)*360
        plt.plot(ra, dec, 'b-')

    plt.axis([360,0,-90,90])


if __name__ == '__main__':
    plt.clf()
    plot_healpix_boundaries(1)
    plt.savefig('hp1.png')

    plt.clf()
    plot_healpix_boundaries(2)
    plt.savefig('hp2.png')
    
