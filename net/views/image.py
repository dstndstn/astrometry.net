import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms
from django.http import HttpResponseRedirect

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from tmpfile import *
from sdss_image import plot_sdss_image

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command

from astrometry.net.views.comment import *

def user_image(req, user_image_id=None):
    image = get_object_or_404(UserImage, pk=user_image_id)

    job = image.get_best_job() 
    calib = None
    if job:
        calib = job.calibration
    comment_form = PartialCommentForm()
    context = {
        'image': image,
        'job': job,
        'calib': calib,
        'comment_form': comment_form,
        'request': req
    }
    return render_to_response('user_image.html', context,
        context_instance = RequestContext(req))

def serve_image(req, id=None):
    image = get_object_or_404(Image, pk=id)
    df = image.disk_file
    imgfn = df.get_path()
    f = open(imgfn)
    res = HttpResponse(f)
    res['Content-type'] = image.get_mime_type()
    return res

def annotated_image(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    ui = job.user_image
    img = ui.image
    df = img.disk_file
    imgfn = df.get_path()
    wcsfn = job.get_wcs_file()
    pnmfn = get_temp_file()
    (filetype, errstr) = image2pnm.image2pnm(imgfn, pnmfn)
    if errstr:
        logmsg('Error converting image file %s: %s' % (imgfn, errstr))
        return HttpResponse('plot failed')
    annfn = get_temp_file()
    cmd = 'plot-constellations -w %s -i %s -o %s -N -C -B' % (wcsfn, pnmfn, annfn)
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('plot failed')
    f = open(annfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

def onthesky_image(req, calid=None):
    from astrometry.net.views.onthesky import plot_aitoff_wcs_outline
    from astrometry.util import util as anutil
    cal = get_object_or_404(Calibration, pk=calid)
    wcsfn = cal.get_wcs_file()
    plotfn = get_temp_file()
    #
    wcs = anutil.Tan(wcsfn, 0)
    zoom = wcs.radius() < 15.
    plot_aitoff_wcs_outline(wcsfn, plotfn, zoom=zoom)
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

def onthesky_zoom1_image(req, calid=None):
    from astrometry.net.views.onthesky import plot_wcs_outline
    from astrometry.util import util as anutil
    cal = get_object_or_404(Calibration, pk=calid)
    wcsfn = cal.get_wcs_file()
    plotfn = get_temp_file()
    #
    wcs = anutil.Tan(wcsfn, 0)
    zoom = wcs.radius() < 1.5
    plot_wcs_outline(wcsfn, plotfn, zoom=zoom)
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

def onthesky_zoom2_image(req, calid=None):
    from astrometry.net.views.onthesky import plot_wcs_outline
    from astrometry.util import util as anutil
    cal = get_object_or_404(Calibration, pk=calid)
    wcsfn = cal.get_wcs_file()
    plotfn = get_temp_file()
    wcs = anutil.Tan(wcsfn, 0)
    zoom = wcs.radius() < 0.15
    plot_wcs_outline(wcsfn, plotfn, width=3.6, grid=1, zoom=zoom, zoomwidth=0.36)
    #hd=True is too cluttered at this level
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

def galex_image(req, calid=None):
    from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    from astrometry.net.galex_jpegs import plot_into_wcs

    cal = get_object_or_404(Calibration, pk=calid)
    wcsfn = cal.get_wcs_file()
    plotfn = get_temp_file()
    #
    plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR)
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res


def sdss_image(req, calid=None):
    cal = get_object_or_404(Calibration, pk=calid)
    wcsfn = cal.get_wcs_file()

    plotfn = get_temp_file()
    #
    plot_sdss_image(wcsfn, plotfn)
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

# 2MASS:
# Has a SIA service, but does not make mosaics.
# Documented here:
# http://irsa.ipac.caltech.edu/applications/2MASS/IM/docs/siahelp.html
# Eg:
# http://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?POS=187.03125,16.9577633&SIZE=1&type=ql

# WISE:
# -IPAC/IRSA claims to have VO services
# -elaborate javascripty interface


def image_set(req, category, id):
    default_category = 'user'
    cat_classes = {
        'user':User,
        'album':Album,
        'tag':Tag,
    }

    if category not in cat_classes:
        category = default_category

    cat_class = cat_classes[category]
    cat_obj = get_object_or_404(cat_class, pk=id)
    
    set_names = {
        'user':'Submitted by User %s' % cat_obj.pk,
        'album':'Album: %s' % cat_obj.pk,
        'tag':'Tag: %s' % cat_obj.pk,
    } 
    image_set_title = set_names[category]

    context = {
        'images': cat_obj.user_images.all,
        'image_set_title':image_set_title,
    }
   
    return render_to_response('image_set.html',
        context,
        context_instance = RequestContext(req))
