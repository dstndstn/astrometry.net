import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
import settings

from astrometry.net.models import *
from django.contrib.auth.models import User

from astrometry.util.util import *
from astrometry.util.starutil_numpy import *

import logging
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)

# How many pixels per healpix tile?
npix = 1000
# What is the pixel scale of the top-level healpixes?
topscale = np.sqrt((4. * np.pi * (180/np.pi)**2 * 3600.**2) / (12 * npix**2))
print 'Top-level scale:', topscale

level = 9
nside = 2**level
print 'nside', nside
print 'n pix', 12 * nside**2

cals = Calibration.objects.all()
print 'Calibrations:', cals.count()
cals = cals.select_related('raw_tan')

pixscales = []
calids = []
for cal in cals:
    pixscales.append(cal.raw_tan.get_pixscale())
    calids.append(cal.id)
pixscale = np.array(pixscales)
calids = np.array(calids)

plt.clf()
plt.hist(np.log2(pixscales), 50)
for scale in range(10):
    plt.axvline(np.log2(topscale / (2.**scale)))
plt.savefig('scales.png')

slo = topscale / 2.**(level+0.5)
shi = topscale / 2.**(level-0.5)
I = np.flatnonzero((pixscales >= slo) * (pixscales < shi))
print len(I), 'in range', slo, shi

#polys = []
ra,dec = [],[]
for i in I:
    tan = cals[i].raw_tan
    ra.append(tan.crval1)
    dec.append(tan.crval2)
ra = np.array(ra)
dec = np.array(dec)

plt.clf()
plt.plot(ra, dec, 'r.', alpha=0.1)
plt.savefig('rd.png')

r1,d1 = healpix_to_radecdeg(0, nside, 0., 0.)
r2,d2 = healpix_to_radecdeg(0, nside, 0.5, 0.5)
hpradius = degrees_between(r1,d1, r2,d2)
print 'Healpix radius:', hpradius

hpims = {}

for i in I:
    tan = cals[i].raw_tan
    r,d,radius = tan.get_center_radecradius()
    radius = np.hypot(radius, hpradius)
    #hh = healpix_get_neighbours_within_range_radec(r, d, radius, nside)
    hh = healpix_rangesearch_radec(r, d, radius, nside)
    #print 'Healpixes in range:', hh
    for hp in hh:
        if not hp in hpims:
            hpims[hp] = []
        lst = hpims[hp]
        lst.append(i)
        if len(lst) > 1:
            print len(lst), 'in hp', hp, ':', lst

            
