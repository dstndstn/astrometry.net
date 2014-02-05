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

topscale = 256.

def addcal(cal, version, hpmod, hpnum):
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
    #print 'File type:', ft

    # HACK
    #if not 'image' in ft:
    # HACK HACK HACK
    if not 'JPEG' in ft:
        return False

    wcsfn = cal.get_wcs_file()
    #print 'WCS file:', wcsfn
    df = cal.job.user_image.image.disk_file
    #print 'Original filename:', cal.job.user_image.original_file_name
    #print 'Submission:', cal.job.user_image.submission
    #print 'DiskFile', df
    ft = df.file_type
    fn = df.get_path()
    if 'JPEG' in ft:
        print 'Reading', fn
        I = plt.imread(fn)
        print 'Read', I.shape, I.dtype
        if len(I.shape) == 2:
            I = I[:,:,np.newaxis].repeat(3, axis=2)
        assert(len(I.shape) == 3)
        u = np.unique(I.ravel())
        print 'Number of unique pixel values:', len(u)
        if I.dtype != np.uint8:
            #print 'Datatype:', I.dtype
            return False
        # arbitrary value!
        #if len(u) <= 25:
        #    continue
    wcs = Sip(wcsfn)
    #print 'WCS', wcs

    print 'Pixscale', tan.get_pixscale()
    nside = 2 ** int(np.round(np.log2(topscale / tan.get_pixscale())))
    print 'Nside', nside
    nside = np.clip(nside, 1, 2**10)
    print 'Nside', nside

    r1,d1 = healpix_to_radecdeg(0, nside, 0., 0.)
    r2,d2 = healpix_to_radecdeg(0, nside, 0.5, 0.5)
    hpradius = degrees_between(r1,d1, r2,d2)
    # HACK -- padding for squished parallelograms
    hpradius *= 1.5

    r,d,radius = tan.get_center_radecradius()
    radius = np.hypot(radius, hpradius)
    hh = healpix_rangesearch_radec(r, d, radius, nside)
    hh.sort()
    print 'Healpixes:', hh

    if hpmod:
        hh = [h for h in hh if h % hpmod == hpnum]
        print 'Cut to healpixes:', hh

    for hp in hh:
        print 'Healpix', hp
        # Check for actual overlap before (possibly) creating EnhancedImage
        hpwcs,nil = EnhancedImage.get_healpix_wcs(nside, hp, topscale)
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(hpwcs, wcs, [], 3)
        except NoOverlapError:
            print 'No actual overlap'
            continue
        print len(Yo), 'resampled pixels'
        if len(Yo) == 0:
            continue

        en,created = EnhancedImage.objects.get_or_create(
            version=version, nside=nside, healpix=hp)
            
        if created or en.wcs is None:
            if created:
                # print 'No EnhancedImage for this nside/healpix yet'
                pass
            else:
                # print 'Re-initializing', en
                pass
            en.init()
            en.save()
        else:
            #print 'EnhancedImage exists:', en
            try:
                #print 'Cals:', en.cals.all()
                en.cals.get(id=cal.id)
                print 'This calibration has already been added to this EnhancedImage'
                continue
            except:
                #print 'Checking whether this cal has been added to this EnhancedImage:'
                #import traceback
                #traceback.print_exc()
                pass

        hpwcs = en.wcs.to_tanwcs()

        enhI,enhW = en.read_files()
        enhM = (enhW > 0)

        # Cut to pixels within healpix
        K = enhM[Yo, Xo]
        Xo,Yo = Xo[K],Yo[K]
        Xi,Yi = Xi[K],Yi[K]
        #print len(Yo), 'resampled within healpix'
        if len(Yo) == 0:
            continue

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

        en.write_files(enhI, enhW)
        en.maxweight = enhW.max()

        en.cals.add(cal)
        #print 'Cals in this EnhancedImage:', en.cals
        en.save()



if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%(prog)')
    parser.add_option('-v', dest='verbose', default=False, action='store_true')
    parser.add_option('-D', dest='delete', action='store_true',
                      help='Delete all database entries before beginning?')

    parser.add_option('--mod', dest='mod', type=int, default=0,
                      help='Split processing into this many pieces')
    parser.add_option('--num', dest='num', type=int,
                      help='Process this number out of "--mod" pieces (0-indexed)')

    opt,args = parser.parse_args()

    if opt.mod:
        assert(opt.num < opt.mod)

    lvl = logging.INFO
    if opt.verbose:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    
    if opt.delete:
        print 'DELETING ALL'
        EnhanceVersion.objects.all().delete()
        EnhancedImage.objects.all().delete()
        print 'ok'

    enver,created = EnhanceVersion.objects.get_or_create(name='v1', topscale=topscale)
    print 'Version:', enver

    cals = Calibration.objects.all()
    print 'Calibrations:', cals.count()
    cals = cals.select_related('raw_tan')

    # level = 9
    # slo = topscale / 2.**(level+0.5)
    # shi = topscale / 2.**(level-0.5)

    ncals = cals.count()
    for ical in range(ncals):
        print
        print 'Calibration', ical, 'of', ncals
        cal = cals[ical]
        print 'Cal', cal

        # pixscale = cal.raw_tan.get_pixscale()
        # if pixscale < slo or pixscale > shi:
        #     print 'Skipping: pixscale', pixscale
        #     continue

        addcal(cal, enver, opt.mod, opt.num)

    sys.exit(0)

'''
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

'''
