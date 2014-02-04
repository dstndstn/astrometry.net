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
from astrometry.util.resample import *
from astrometry.util.plotutils import *

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





def runcals(hp, calI):
    xyz0 = np.array(healpix_to_xyz(hp, nside, 0., 0.))
    xyz1 = np.array(healpix_to_xyz(hp, nside, 1., 0.))
    xyz2 = np.array(healpix_to_xyz(hp, nside, 0., 1.))
    xyz3 = np.array(healpix_to_xyz(hp, nside, 1., 1.))
    
    d1 = xyz1 - xyz0
    #d2 = xyz2 - xyz0
    d2 = np.cross(xyz0, d1)
    print 'd1', d1
    print 'xyz2-xyz0:', xyz2-xyz0
    print 'd2', d2
    
    d3 = xyz3 - xyz0
    # this should be only different in z
    print 'd3', d3
    
    d2 /= np.sqrt(np.sum(d2**2))
    d3 /= np.sqrt(np.sum(d3**2))
    
    print 'dot', np.sum(d2 * d3)
    theta = np.arccos(np.sum(d2*d3))
    #theta = np.deg2rad(45.)
    print 'theta', np.rad2deg(theta)
    
    r,d = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
    
    pscale = topscale / 2**level
    print 'pixscale', pscale
    pscale /= 3600.
    
    wcs = Tan(r, d, 0., 0.,
              pscale * np.cos(theta), -pscale * np.sin(theta),
              pscale * np.sin(theta),  pscale * np.sin(theta),
              1000,1000)
    
    xy = np.array([wcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                   [xyz0, xyz1, xyz2, xyz3]])
    # drop ok
    xy = xy[:,1:]
    print 'xy', xy
    xlo,ylo = xy.min(axis=0)
    xhi,yhi = xy.max(axis=0)
    
    W,H = int(np.ceil(xhi - xlo)), int(np.ceil(yhi - ylo))
    hpwcs = Tan(r, d, 1.-xlo, 1.-ylo,
                pscale * np.cos(theta), -pscale * np.sin(theta),
                pscale * np.sin(theta),  pscale * np.sin(theta),
                float(W), float(H))
    
    xy = np.array([hpwcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                   [xyz0, xyz1, xyz2, xyz3]])
    print 'xy', xy
    
    # plt.clf()
    # plt.plot(xy[:,1], xy[:,2], 'r.-')
    # plt.axis('scaled')
    # plt.savefig('hp.png')
    
    ############################

    enhI = np.zeros((H,W), np.float32)
    enhN = np.zeros((H,W), np.int32)
    
    for cali in calI:
        cal = cals[cali]
        wcsfn = cal.get_wcs_file()
        print 'WCS file:', wcsfn
        df = cal.job.user_image.image.disk_file
        print 'DiskFile', df
        ft = df.file_type
        fn = df.get_path()
        if 'JPEG' in ft:
            I = plt.imread(fn)
            print 'Read', I.shape, I.dtype
        else:
            print 'Filetype', ft
            continue

        wcs = Sip(wcsfn)
        print 'WCS', wcs
    
        #try:
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(hpwcs, wcs, [], 3)

        # for plane in rgb:
        plane = 0

        if I.dtype == np.uint8:
            data = (I[:,:,plane] / 255.).astype(np.float32)
        else:
            continue
        data += np.random.uniform(1./255, size=data.shape)

        #resam = np.zeros((H,W), np.float32)
        #resam[Yo,Xo] = data[Yi,Xi]

        img = data[Yi, Xi]

        #mask = np.zeros((H,W), bool)
        #mask[Yo, Xo] = True

        enh = enhI[Yo, Xo]

        II = np.argsort(img)
        rankimg = np.empty_like(II)
        rankimg[II] = np.arange(len(II))

        EI = np.argsort(enh)
        rankenh = np.empty_like(EI)
        rankenh[EI] = np.arange(len(EI))

        wenh = enhN[Yo, Xo]

        weightFactor = 2.

        rank = ( ((rankenh * wenh) + (rankimg * weightFactor))
                 / (wenh + weightFactor) )
        II = np.argsort(rank)
        rankC = np.empty_like(II)
        rankC[II] = np.arange(len(II))

        Enew = enh[EI[rankC]]
        enhI[Yo,Xo] = Enew
        enhN[Yo,Xo] += 1

        resam = np.zeros((H,W), np.float32)
        resam[Yo,Xo] = img

        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(I, interpolation='nearest', origin='lower')
        plt.subplot(2,2,2)
        plt.imshow(resam, interpolation='nearest', origin='lower')
        plt.subplot(2,2,3)
        plt.imshow(enhN, interpolation='nearest', origin='lower')
        plt.subplot(2,2,4)
        plt.imshow(enhI, interpolation='nearest', origin='lower')
        ps.savefig()


cals = Calibration.objects.all()
print 'Calibrations:', cals.count()
cals = cals.select_related('raw_tan')

hp = 239096
calI = [1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1497, 1498, 1499, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1598, 1600, 1606, 1647, 1657, 1694, 1695, 1697, 1698, 1699, 1700, 1701, 1702, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1713, 1714, 1715, 1716, 1717, 1718]

hp = 2134916
calI = [5336, 5348, 5759, 5778, 5787, 5796, 5806, 5841, 5853, 5863]

ps = PlotSequence('e')
runcals(hp, calI)
sys.exit(0)



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

gotlist = None

for i in I:
    tan = cals[i].raw_tan

    try:
        if not cals[i].job.user_image.publicly_visible:
            print 'Image not public'
            continue
    except:
        print 'Error querying publicly_visible:'
        import traceback
        traceback.print_exc()
        continue

    ft = cals[i].job.user_image.image.disk_file.file_type
    print 'File type:', ft
    # HACK
    #if not 'image' in ft:
    # HACK HACK HACK
    if not 'JPEG' in ft:
        continue

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

        if len(lst) >= 10:
            gotlist = (hp, lst)
            break
    if gotlist:
        break


(hp,calI) = gotlist
print 'Healpix:', hp
print 'Cals:', calI

runcals(hp, calI)

