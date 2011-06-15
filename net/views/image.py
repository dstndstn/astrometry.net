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

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command

def user_image(req, user_image_id=None):
    image = get_object_or_404(UserImage, pk=user_image_id)

    job = image.get_best_job() 
    calib = None
    if job:
        calib = job.calibration
        
    context = {'image': image,
               'job': job,
               'calib': calib,}
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
    f,pnmfn = tempfile.mkstemp()
    os.close(f)
    (filetype, errstr) = image2pnm.image2pnm(imgfn, pnmfn)
    if errstr:
        logmsg('Error converting image file %s: %s' % (imgfn, errstr))
        return HttpResponse('plot failed')
    f,annfn = tempfile.mkstemp()
    os.close(f)
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

def galex_image(req, jobid=None):
    from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    from astrometry.net.galex_jpegs import plot_into_wcs

    job = get_object_or_404(Job, pk=jobid)
    wcsfn = job.get_wcs_file()
    f,plotfn = tempfile.mkstemp()
    os.close(f)
    #
    plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR)
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res


def sdss_image(req, jobid=None):
    from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps

    job = get_object_or_404(Job, pk=jobid)
    wcsfn = job.get_wcs_file()
    f,plotfn = tempfile.mkstemp()
    os.close(f)
    # Parse the wcs.fits file
    wcs = anutil.Tan(wcsfn, 0)
    # arcsec radius
    #scale = math.hypot(wcs.imagew, wcs.imageh)/2. * wcs.pixel_scale()
    # grab SDSS tiles with about the same resolution as this image.
    logmsg('Image scale is', wcs.pixel_scale(), 'arcsec/pix')
    # size of SDSS image tiles to request, in pixels
    sdsssize = 512
    scale = sdsssize * wcs.pixel_scale() / 60.
    # healpix-vs-north-up rotation
    nside = anutil.healpix_nside_for_side_length_arcmin(scale / math.sqrt(2.))
    nside = 2 ** int(math.ceil(math.log(nside)/math.log(2.)))
    logmsg('Next power-of-2 nside:', nside)
    ra,dec = wcs.radec_center()
    logmsg('Image center is RA,Dec', ra, dec)

    dirnm = os.path.join(settings.SDSS_TILE_DIR, 'nside%i'%nside)
    if not os.path.exists(dirnm):
        os.makedirs(dirnm)

    #hp = anutil.radecdegtohealpix(ra, dec, nside)
    #logmsg('Healpix of center:', hp)
    radius = wcs.radius()
    hps = anutil.healpix_rangesearch_radec(ra, dec, radius, nside)
    logmsg('Healpixes in range:', hps)

    scale = math.sqrt(2.) * anutil.healpix_side_length_arcmin(nside) * 60. / float(sdsssize)
    logmsg('Grabbing SDSS tile with scale', scale, 'arcsec/pix')

    plot = ps.Plotstuff(outformat='png', wcsfn=wcsfn)
    img = plot.image
    img.format = ps.PLOTSTUFF_FORMAT_JPG
    img.resample = 1

    for hp in hps:
        fn = os.path.join(dirnm, '%i.jpg'%hp)
        logmsg('Checking for filename', fn)
        if not os.path.exists(fn):
            ra,dec = anutil.healpix_to_radecdeg(hp, nside, 0.5, 0.5)
            logmsg('Healpix center is RA,Dec', ra, dec)
            url = ('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?' +
                   'ra=%f&dec=%f&scale=%f&opt=&width=%i&height=%i' %
                   (ra, dec, scale, sdsssize, sdsssize))
            urllib.urlretrieve(url, fn)
            logmsg('Wrote', fn)
        swcsfn = os.path.join(dirnm, '%i.wcs'%hp)
        logmsg('Checking for WCS', swcsfn)
        if not os.path.exists(swcsfn):
            # Create WCS header
            cd = scale / 3600.
            swcs = anutil.Tan(ra, dec, sdsssize/2 + 0.5, sdsssize/2 + 0.5,
                              -cd, 0, 0, -cd, sdsssize, sdsssize)
            swcs.write_to(swcsfn)
            logmsg('Wrote WCS to', swcsfn)

        img.set_wcs_file(swcsfn, 0)
        img.set_file(fn)
        plot.plot('image')

    if False:
        out = plot.outline
        plot.color = 'white'
        plot.alpha = 0.25
        for hp in hps:
            swcsfn = os.path.join(dirnm, '%i.wcs'%hp)
            ps.plot_outline_set_wcs_file(out, swcsfn, 0)
            plot.plot('outline')

    plot.write(plotfn)
    
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
