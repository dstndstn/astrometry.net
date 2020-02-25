from __future__ import print_function
import pylab as plt
import numpy as np

from django.http import HttpResponse
from django.shortcuts import get_object_or_404

from astrometry.net.models import *
from astrometry.util.resample import *
from astrometry.net.tmpfile import *

def simple_histeq(pixels, getinverse=False, mx=256):
    assert(pixels.dtype in [np.uint8, np.uint16])

    if not getinverse:
        h = np.bincount(pixels, minlength=mx)
        # pixel value -> quantile map.
        # If you imagine jittering the pixels so there are no repeats,
        # this assigns the middle quantile to a pixel value.
        quant = h * 0.5
        cs = np.cumsum(h)
        quant[1:] += cs[:-1]
        quant /= float(cs[-1])
        # quant = np.cumsum(h / float(h.sum()))
        return quant[pixels]

    # This inverse function has slightly weird properties -- it
    # puts a ramp across each pixel value, so inv(0.) may produce
    # values as small as -0.5, and inv(1.) may produce 255.5

    h = np.bincount(pixels.astype(int)+1, minlength=mx+1)
    quant = h[1:] * 0.5
    cs = np.cumsum(h)
    quant[1:] += cs[1:-1]
    quant /= float(cs[-1])

    # interp1d is fragile -- remove duplicate "yy" values that
    # otherwise cause nans.
    yy = cs / float(cs[-1])
    xx = np.arange(mx + 1) - 0.5
    I = np.append([0], 1 + np.flatnonzero(np.diff(yy)))
    print('mx:', mx)
    print('xx:', len(xx))
    print('yy:', len(yy))
    print('I:', I.min(), I.max())
    yy = yy[I]
    xx = xx[I]
    xx[-1] = mx-0.5
    # print 'yy', yy[0], yy[-1]
    # print 'xx', xx[0], xx[-1]
    inv = interp1d(yy, xx, kind='linear')
    return quant[pixels], inv



def enhanced_ui(req, user_image_id=None):
    ui = UserImage.objects.get(id=user_image_id)
    job = ui.get_best_job()
    return enhanced_image(req, job_id=job.id, size='display')

def enhanced_image(req, job_id=None, size=None):
    job = get_object_or_404(Job, pk=job_id)
    ui = job.user_image
    cal = job.calibration
    tan = cal.raw_tan
    nside,hh = get_healpixes_touching_wcs(tan)
    tt = 'hello %s, job %s, nside %s, hh %s' % (ui, job, nside, hh)

    ver = EnhanceVersion.objects.get(name='v4')
    print('Using', ver)

    EIms = EnhancedImage.objects.filter(version=ver)

    ens = []
    for hp in hh:
        en = EIms.filter(nside=nside, healpix=hp, version=ver)
        if len(en):
            ens.extend(list(en))

    for dnside in range(1, 3):
        if len(ens) == 0:
            bignside = nside / (2**dnside)
            nil,hh = get_healpixes_touching_wcs(tan, nside=bignside)
            tt += 'bigger healpixes: %s: %s' % (bignside, hh)
            for hp in hh:
                en = EIms.filter(nside=bignside, healpix=hp)
                if len(en):
                    ens.extend(list(en))

    tt = tt + ', EnhancedImages: ' + ', '.join('%s'%e for e in ens)

    img = ui.image
    W,H = img.width, img.height

    tt = tt + 'image size %ix%i' % (W,H)

    #return HttpResponse(tt)

    targetwcs = tan.to_tanwcs()
    #print 'Target WCS:', targetwcs
    #print 'W,H', W,H
    logmsg('wcs:', str(targetwcs))

    if size == 'display':
        scale = float(ui.image.get_display_image().width)/ui.image.width
        logmsg('scaling:', scale)
        targetwcs = targetwcs.scale(scale)
        logmsg('scaled wcs:', str(targetwcs))
        H,W = targetwcs.get_height(), targetwcs.get_width()
        img = ui.image.get_display_image()

    print(tt)
    ee = np.zeros((H,W,3), np.float32)

    imgdata = None
    df = img.disk_file
    ft = df.file_type
    fn = df.get_path()
    if 'JPEG' in ft:
        print('Reading', fn)
        I = plt.imread(fn)
        print('Read', I.shape, I.dtype)
        if len(I.shape) == 2:
            I = I[:,:,np.newaxis].repeat(3, axis=2)
        assert(len(I.shape) == 3)
        if I.shape[2] > 3:
            I = I.shape[:,:,:3]
        # vertical FLIP to match WCS
        I = I[::-1,:,:]
        imgdata = I
        mapped = np.zeros_like(imgdata)

    for en in ens:
        logmsg('Resampling %s' % en)
        wcs = en.wcs.to_tanwcs()
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, wcs, [], 3)
        except OverlapError:
            continue
        #logmsg(len(Yo), 'pixels')
        enI,enW = en.read_files()
        #print 'Cals included in this Enhanced image:'
        #for c in en.cals.all():
        #    print '  ', c
        #logmsg('en:', enI.min(), enI.max())

        if imgdata is not None:
            mask = (enW[Yi,Xi] > 0)
        for b in range(3):
            enI[:,:,b] /= enI[:,:,b].max()
            if imgdata is not None:
                idata = imgdata[Yo[mask],Xo[mask],b]
                DI = np.argsort((idata + np.random.uniform(size=idata.shape))/255.)

                EI = np.argsort(enI[Yi[mask], Xi[mask], b])
                Erank = np.zeros_like(EI)
                Erank[EI] = np.arange(len(Erank))

                mapped[Yo[mask],Xo[mask],b] = idata[DI[Erank]]
            else:
                # Might have to average the coverage here...
                ee[Yo,Xo,b] += enI[Yi,Xi,b]
                # ee[Yo[mask],Xo[mask],b] += enI[Yi[mask],Xi[mask],b]

    tempfn = get_temp_file(suffix='.png')

    if imgdata is not None:
        im = mapped
    else:
        im = np.clip(ee, 0., 1.)
    dpi = 100
    figsize = [x / float(dpi) for x in im.shape[:2][::-1]]
    plt.figure(figsize=figsize, frameon=False, dpi=dpi)
    plt.clf()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(im, interpolation='nearest')

    # rdfn = job.get_rdls_file()
    # rd = fits_table(rdfn)
    # ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
    # plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)

    plt.savefig(tempfn)

    print('Wrote', tempfn)
    f = open(tempfn, 'rb')
    res = HttpResponse(f)
    res['Content-Type'] = 'image/png'
    return res

