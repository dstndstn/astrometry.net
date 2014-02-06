import pylab as plt
import numpy as np

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext, loader

from astrometry.net.models import *
from astrometry.util.resample import *
from astrometry.net.tmpfile import *

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

    ver = EnhanceVersion.objects.get(name='v2')
    print 'Using', ver

    EIms = EnhancedImage.objects.filter(version=ver)

    ens = []
    for hp in hh:
        en = EIms.filter(nside=nside, healpix=hp, version=ver)
        if len(en):
            ens.extent(list(en))

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
    print 'Target WCS:', targetwcs
    print 'W,H', W,H

    if size == 'display':
        image = ui
        scale = float(image.image.get_display_image().width)/image.image.width
        targetwcs = targetwcs.scale(scale)
        print 'Scaled to:', targetwcs
        H,W = targetwcs.get_height(), targetwcs.get_width()

    print tt
    ee = np.zeros((H,W,3), np.float32)

    for en in ens:
        logmsg('Resampling %s' % en)
        wcs = en.wcs.to_tanwcs()
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, wcs, [], 3)
        except OverlapError:
            continue
        print len(Yo), 'pixels'
        enI,enW = en.read_files()

        print 'Cals included in this Enhanced image:'
        for c in en.cals.all():
            print '  ', c

        print 'en:', enI.min(), enI.max()

        #mask = enW[Yi,Xi]
        #Yo = Yo[mask]
        #Xo = Xo[mask]
        # Might have to average the coverage here...
        for b in range(3):
            enI[:,:,b] /= enI[:,:,b].max()
            ee[Yo,Xo,b] += enI[Yi,Xi,b]

    print 'ee', ee.min(), ee.max()

    tempfn = get_temp_file(suffix='.png')

    im = np.clip(ee, 0., 1.)
    print 'im', im.shape, im.dtype

    #plt.imsave(tempfn, im, origin='lower')

    dpi = 100
    figsize = [x / float(dpi) for x in im.shape[:2][::-1]]
    fig = plt.figure(figsize=figsize, frameon=False, dpi=dpi)
    plt.clf()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(im, interpolation='nearest', origin='lower')
    plt.savefig(tempfn)

    print 'Wrote', tempfn
    f = open(tempfn)
    res = HttpResponse(f)
    res['Content-Type'] = 'image/png'
    return res

