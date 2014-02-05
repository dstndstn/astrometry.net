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

from astrometry.net.enhance_models import *

from astrometry.util.util import *
from astrometry.util.starutil_numpy import *
from astrometry.util.resample import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *

import logging
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)

# # Approximately how many pixels per healpix tile?
# npix = 1000
# # What is the pixel scale of the top-level healpixes?
# topscale = np.sqrt((4. * np.pi * (180/np.pi)**2 * 3600.**2) / (12 * npix**2))

topscale = 256.
print 'Top-level scale:', topscale

level = 9
nside = 2**level
print 'nside', nside
print 'n pix', 12 * nside**2

def addcal(cal, version):
    tan = cal.raw_tan
    try:
        if not cal.job.user_image.publicly_visible:
            print 'Image not public'
            return False
    except:
        print 'Error querying publicly_visible:'
        import traceback
        traceback.print_exc()
        return False

    ft = cal.job.user_image.image.disk_file.file_type
    print 'File type:', ft

    # HACK
    #if not 'image' in ft:
    # HACK HACK HACK
    if not 'JPEG' in ft:
        return False

    wcsfn = cal.get_wcs_file()
    print 'WCS file:', wcsfn
    df = cal.job.user_image.image.disk_file
    print 'Original filename:', cal.job.user_image.original_file_name
    print 'Submission:', cal.job.user_image.submission
    print 'DiskFile', df
    ft = df.file_type
    fn = df.get_path()
    if 'JPEG' in ft:
        print 'Reading', fn
        I = plt.imread(fn)
        print 'Read', I.shape, I.dtype
        assert(len(I.shape) == 3)
        u = np.unique(I.ravel())
        print 'Number of unique pixel values:', len(u)
        if I.dtype != np.uint8:
            print 'Datatype:', I.dtype
            return False
        # arbitrary value!
        #if len(u) <= 25:
        #    continue
    wcs = Sip(wcsfn)
    print 'WCS', wcs

    print 'Pixscale', tan.get_pixscale()
    nside = 2 ** int(np.round(np.log2(topscale / tan.get_pixscale())))
    print 'Nside', nside

    r1,d1 = healpix_to_radecdeg(0, nside, 0., 0.)
    r2,d2 = healpix_to_radecdeg(0, nside, 0.5, 0.5)
    hpradius = degrees_between(r1,d1, r2,d2)
    # HACK -- padding for squished parallelograms
    hpradius *= 1.5

    r,d,radius = tan.get_center_radecradius()
    radius = np.hypot(radius, hpradius)
    hh = healpix_rangesearch_radec(r, d, radius, nside)
    #print 'Healpixes in range:', hh
    for hp in hh:

        en,created = EnhancedImage.objects.get_or_create(
            version=version, nside=nside, healpix=hp)
            
        if created:
            print 'No EnhancedImage for this nside/healpix yet'
            en.init()
            en.save()
        else:
            print 'EnhancedImage exists:', en
            if cal in en.cals:
                print 'This calibration has already been added to this EnhancedImage'
                continue

        hpwcs = en.wcs.to_tanwcs()
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(hpwcs, wcs, [], 3)

        enhI = fitsio.read(en.get_image_path())
        enhW = fitsio.read(en.get_weight_path())
        enhM = (enhW > 0)

        # Cut to pixels within healpix
        print len(Yo), 'resampled pixels'
        K = enhM[Yo, Xo]
        Xo,Yo = Xo[K],Yo[K]
        Xi,Yi = Xi[K],Yi[K]
        print len(Yo), 'resampled within healpix'

        assert(len(enhI.shape) == 3)
        # RGB
        assert(enhI.shape[2] == 3)
        assert(I.shape[2] == 3)

        for b in range(3):
            data = (I[:,:,b] / 255.).astype(np.float32)
            data += np.random.uniform(1./255, size=data.shape)

            img = data[Yi, Xi]
            enh = enhI[Yo, Xo, b]
            wenh = enhW[Yo, Xo]

            II = np.argsort(img)
            rankimg = np.empty_like(II)
            rankimg[II] = np.arange(len(II))

            EI = np.argsort(enh)
            rankenh = np.empty_like(EI)
            rankenh[EI] = np.arange(len(EI))

            weightFactor = 2.

            rank = ( ((rankenh * wenh) + (rankimg * weightFactor))
                     / (wenh + weightFactor) )
            II = np.argsort(rank)
            rankC = np.empty_like(II)
            rankC[II] = np.arange(len(II))

            Enew = enh[EI[rankC]]
            enhI[Yo,Xo, b] = Enew

        enhW[Yo,Xo] += 1.

        en.cals.add(cal)
        en.save()


enver,created = EnhanceVersion.objects.get_or_create(name='v1', topscale=topscale)
print 'Created', enver

cals = Calibration.objects.all()
print 'Calibrations:', cals.count()
cals = cals.select_related('raw_tan')

slo = topscale / 2.**(level+0.5)
shi = topscale / 2.**(level-0.5)

ncals = cals.count()
for ical in range(ncals):
    cal = cals[i]
    print 'Cal', cal
    pixscale = cal.raw_tan.get_pixel_scale()
    if pixscale < slo or pixscale > shi:
        print 'Skipping: pixscale', pixscale
        continue

    addcal(cal, enver)
    

sys.exit(0)













def runcals(hp, calI, plots):
    xyz0 = np.array(healpix_to_xyz(hp, nside, 0., 0.))
    xyz1 = np.array(healpix_to_xyz(hp, nside, 1., 0.))
    xyz2 = np.array(healpix_to_xyz(hp, nside, 0., 1.))
    xyz3 = np.array(healpix_to_xyz(hp, nside, 1., 1.))

    r0,d0 = xyztoradec(xyz0)
    up = np.array([-np.sin(np.deg2rad(d0))*np.cos(deg2rad(r0)),
                   -np.sin(np.deg2rad(d0))*np.sin(deg2rad(r0)),
                    np.cos(np.deg2rad(d0))])

    if plots:

        plt.clf()
        rd = []
        for xyz in [xyz0, xyz1, xyz3, xyz2, xyz0]:
            rd.append(xyztoradec(xyz))
        rd = np.array(rd)
        plt.plot(rd[:,0], rd[:,1], 'b.-')
        plt.text(rd[0,0], rd[0,1], '0,0')
        plt.text(rd[3,0], rd[3,1], '0,1')
        plt.text(rd[1,0], rd[1,1], '1,0')
        plt.text(rd[2,0], rd[2,1], '1,1')
        r0,d0 = rd.min(axis=0)
        r1,d1 = rd.max(axis=0)
        setRadecAxes(r0,r1, d0,d1)
        ps.savefig()
    
    d1 = xyz1 - xyz0
    d1 /= np.sqrt(np.sum(d1**2))
    #print 'd1', d1
    #print 'xyz dot up', np.sum(xyz0 * up)
    theta = np.pi/2. - np.arccos(np.sum(d1 * up))
    #print 'theta', np.rad2deg(theta)

    r,d = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
    pscale = topscale / 2**level
    #print 'pixscale', pscale
    pscale /= 3600.
    
    wcs = Tan(r, d, 500., 500.,
              pscale * np.cos(theta), -pscale * np.sin(theta),
              pscale * np.sin(theta),  pscale * np.cos(theta),
              1000,1000)

    if plots:
        print 'xyz0', xyz0
        print 'xyz1', xyz1
        print 'xyz2', xyz2
        print 'xyz3', xyz3

        plt.clf()
        # healpix outline
        rd = []
        for xyz in [xyz0, xyz1, xyz3, xyz2, xyz0]:
            rd.append(xyztoradec(xyz))
        rd = np.array(rd)
        plt.plot(rd[:,0], rd[:,1], 'b.-')
        plt.text(rd[0,0], rd[0,1], '0,0')
        plt.text(rd[3,0], rd[3,1], '0,1')
        plt.text(rd[1,0], rd[1,1], '1,0')
        plt.text(rd[2,0], rd[2,1], '1,1')
        # WCS outline
        rd2 = []
        for x,y in [(1,1),(1000,1),(1000,1000), (1,1000)]:
            rd2.append(wcs.pixelxy2radec(x,y))
        rd2 = np.array(rd2)
        plt.plot(rd2[:,0], rd2[:,1], 'r.-')
        # axes
        r0,d0 = rd.min(axis=0) - 0.05
        r1,d1 = rd.max(axis=0) + 0.05
        setRadecAxes(r0,r1, d0,d1)
        ps.savefig()
    
    xy = np.array([wcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                   [xyz0, xyz1, xyz2, xyz3]])
    # drop ok
    xy = xy[:,1:]
    #print 'xy', xy
    xlo,ylo = xy.min(axis=0)
    xhi,yhi = xy.max(axis=0)
    
    # Make the WCS just contain the healpix -- magic 501 here comes from 500 above.
    W,H = int(np.ceil(xhi - xlo)), int(np.ceil(yhi - ylo))
    hpwcs = Tan(r, d, 501.-xlo, 501.-ylo,
                pscale * np.cos(theta), -pscale * np.sin(theta),
                pscale * np.sin(theta),  pscale * np.cos(theta),
                float(W), float(H))

    hpxy = np.array([hpwcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                     [xyz0, xyz1, xyz3, xyz2]])
    hpxy = hpxy[:,1:]
    
    if plots:
        xy = np.array([hpwcs.xyz2pixelxy(xyz[0], xyz[1], xyz[2]) for xyz in
                       [xyz0, xyz1, xyz2, xyz3]])
        xy = xy[:,1:]
        #print 'xy', xy
        #print 'Healpix WCS:', hpwcs
        #print 'Image size', W, H
        plt.clf()
        ii = np.array([0, 1, 3, 2, 0])
        plt.plot(xy[ii,0], xy[ii,1], 'r.-')
        plt.axis('scaled')
        ps.savefig()
    

    enhI = np.zeros((H,W), np.float32)
    enhW = np.zeros((H,W), np.float32)
    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    # inside-healpix mask.
    enhM = point_in_poly(xx, yy, hpxy-1.)
    del xx
    del yy
    npix = np.sum(enhM)
    enhI[enhM] = np.random.permutation(npix)

    print 'N pix:', W*H
    print 'N pix within healpix:', np.sum(enhM)
    
    for cali in calI:
        cal = cals[cali]
        wcsfn = cal.get_wcs_file()
        print 'WCS file:', wcsfn
        df = cal.job.user_image.image.disk_file
        print 'Original filename:', cal.job.user_image.original_file_name
        print 'Submission:', cal.job.user_image.submission
        print 'DiskFile', df
        ft = df.file_type
        fn = df.get_path()
        if 'JPEG' in ft:
            print 'Reading', fn
            I = plt.imread(fn)
            print 'Read', I.shape, I.dtype
            u = np.unique(I.ravel())
            print 'Number of unique pixel values:', len(u)
            ## arbitrary value
            #if len(u) <= 25:
            #    continue
        else:
            print 'Filetype', ft
            continue

        wcs = Sip(wcsfn)
        print 'WCS', wcs
    
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(hpwcs, wcs, [], 3)

        # for plane in rgb:
        plane = 0

        if I.dtype == np.uint8:
            data = (I[:,:,plane] / 255.).astype(np.float32)
        else:
            continue
        data += np.random.uniform(1./255, size=data.shape)

        # Cut to pixels within healpix
        print len(Yo), 'resampled pixels'
        K = enhM[Yo, Xo]
        Xo,Yo = Xo[K],Yo[K]
        Xi,Yi = Xi[K],Yi[K]
        print len(Yo), 'resampled within healpix'

        img = data[Yi, Xi]
        enh = enhI[Yo, Xo]
        wenh = enhW[Yo, Xo]

        II = np.argsort(img)
        rankimg = np.empty_like(II)
        rankimg[II] = np.arange(len(II))

        EI = np.argsort(enh)
        rankenh = np.empty_like(EI)
        rankenh[EI] = np.arange(len(EI))

        weightFactor = 2.

        rank = ( ((rankenh * wenh) + (rankimg * weightFactor))
                 / (wenh + weightFactor) )
        II = np.argsort(rank)
        rankC = np.empty_like(II)
        rankC[II] = np.arange(len(II))

        Enew = enh[EI[rankC]]
        enhI[Yo,Xo] = Enew
        enhW[Yo,Xo] += 1.

        resam = np.zeros((H,W), np.float32)
        resam[Yo,Xo] = img

        Ilo,Ihi = [np.percentile(I, p) for p in [5, 95]]
        if Ilo == Ihi:
            Ihi = Ilo + 1e-3
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(np.clip((I - Ilo)/float(Ihi-Ilo), 0., 1.),
                   interpolation='nearest', origin='lower')
        plt.title('Image')
        plt.subplot(2,2,2)
        plt.imshow(np.clip((resam - Ilo)/float(Ihi-Ilo), 0., 1.),
                   interpolation='nearest', origin='lower')
        plt.title('Resampled image')
        plt.subplot(2,2,3)
        plt.imshow(enhW, interpolation='nearest', origin='lower')
        plt.title('E Weight')
        plt.subplot(2,2,4)
        plt.imshow(enhI, interpolation='nearest', origin='lower')
        plt.title('E Image')
        ps.savefig()


cals = Calibration.objects.all()
print 'Calibrations:', cals.count()
cals = cals.select_related('raw_tan')

hp = 239096
calI = [1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1497, 1498, 1499, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1598, 1600, 1606, 1647, 1657, 1694, 1695, 1697, 1698, 1699, 1700, 1701, 1702, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1713, 1714, 1715, 1716, 1717, 1718]

hp = 2134916
calI = [5336, 5348, 5759, 5778, 5787, 5796, 5806, 5841, 5853, 5863]

ps = PlotSequence('e')
plots = False
runcals(hp, calI, plots)
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

runcals(hp, calI, plots)

