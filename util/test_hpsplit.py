# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.run_command import *
from astrometry.util.plotutils import *

# Create test file with a grid of RA,Dec points.

r = np.arange(0, 360, 1)
d = np.arange(-90, 90.05, 1)

R,D = np.meshgrid(r, d)

T = tabledata()
T.ra  = R.ravel()
T.dec = D.ravel()
fn = 'test-hpsplit-in1.fits'
T.writeto(fn)

splitpat = 'test-hpsplit-1-%02i.fits'
cmd = 'hpsplit -o %s -n 1 -m 10 %s' % (splitpat, fn)
rtn,out,err = run_command(cmd)
assert(rtn == 0)

ps = PlotSequence('test_hpsplit')
for hp in range(12):
    T = fits_table(splitpat % hp)
    print('Healpix', hp, 'got', len(T))
    plt.clf()
    plothist(T.ra, T.dec, 360, range=((0,360),(-90,90)))
    vv = np.linspace(0, 1, 100)
    rd = []
    for v in vv:
        rd.append(healpix_to_radecdeg(hp, 1, 0., v))
    for v in vv:
        rd.append(healpix_to_radecdeg(hp, 1, v, 1.))
    for v in reversed(vv):
        rd.append(healpix_to_radecdeg(hp, 1, 1., v))
    for v in reversed(vv):
        rd.append(healpix_to_radecdeg(hp, 1, v, 0.))
    rd = np.array(rd)
    plt.plot(rd[:,0], rd[:,1], '-', color=(0,0.5,1.), lw=2)
    plt.axis([0, 360, -90, 90])
    ps.savefig()



