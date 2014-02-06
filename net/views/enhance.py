from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext, loader

from astrometry.net.models import *

def enhanced_ui(req, user_image_id=None):
    ui = UserImage.objects.get(id=user_image_id)
    job = ui.get_best_job()
    cal = job.calibration
    tan = cal.raw_tan
    nside,hh = get_healpixes_touching_wcs(tan)
    tt = 'hello %s, job %s, nside %s, hh %s' % (ui, job, nside, hh)

    ens = []
    for hp in hh:
        en = EnhancedImage.objects.filter(nside=nside, healpix=hp)
        if len(en):
            ens.extent(list(en))

    tt = tt + ', EnhancedImages: ' + ', '.join('%s'%e for e in ens)

    return HttpResponse(tt)

