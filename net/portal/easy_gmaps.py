from django.http import *
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse
from django.db.models import Q

import re
import os.path
import os
import popen2
import sha
import logging
import commands
import shutil
import math

from astrometry.net.util.run_command import run_command
import astrometry.net.settings as settings

logfile        = settings.LOGFILE
tilerender     = settings.TILERENDER
cachedir       = settings.CACHEDIR
rendercachedir = settings.RENDERCACHEDIR
tempdir        = settings.TEMPDIR

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=logfile,
                    )
def asinh(x):
    return math.log(x + math.sqrt(1. + x*x))

def getbb(request):
    try:
        bb = request.GET['bb']
    except (KeyError):
        raise KeyError('No bb')
    bbvals = bb.split(',')
    if (len(bbvals) != 4):
        raise KeyError('Bad bb')
    longmin  = float(bbvals[0])
    latmin   = float(bbvals[1])
    longmax  = float(bbvals[2])
    latmax   = float(bbvals[3])

    pi = math.pi

    latminrad = latmin * pi / 180.0
    latmaxrad = latmax * pi / 180.0

    xmin = longmin / 360.0
    xmax = longmax / 360.0
    if xmin < 0:
        xmin += 1.0
        xmax += 1.0
    ymin = 0.5 + (asinh(math.tan(latminrad)) / (2.0 * pi))
    ymax = 0.5 + (asinh(math.tan(latmaxrad)) / (2.0 * pi))

    newymax = 1 - ymin
    ymin = 1 - ymax
    ymax = newymax
    
    return (xmin, xmax, ymin, ymax)

def tile(request):
    #logging.debug('query() starting')
    try:
        (xmin, xmax, ymin, ymax) = getbb(request)
    except KeyError, x:
        return HttpResponse(x)
    try:
        imw = int(request.GET['w'])
        imh = int(request.GET['h'])
        layers = request.GET['layers'].split(',')
    except (KeyError):
        return HttpResponse('No w/h/layers')
    if (imw == 0 or imh == 0):
        return HttpResponse('Bad w or h')
    if (len(layers) == 0):
        return HttpResponse('No layers')

    pnmfile = '/tmp/149.pnm'

    left = xmin * 9904.
    top = ymin * 9904.
    W = (xmax - xmin) * 9904.
    H = (ymax - ymin) * 9904.

    cmdline = 'pnmcut -left %i -top %i -width %i -height %i %s | pnmscale -width=%i -height=%i | pnmtopng' % (left, top, W, H, pnmfile, imw, imh)

    m = sha.new()
    m.update(cmdline)
    digest = m.hexdigest()
    fn = tempdir + '/easy-gmaps-' + digest + '.png'

    if not os.path.exists(fn):
        # Run it!
        cmd = cmdline + ' > ' + fn + ' 2>> ' + logfile
        logging.debug('running: ' + cmd)

        (rtn, stdout, stderr) = run_command(cmd)
        if rtn:
            return HttpResponse('Command failed: ' + cmd + ', stdout=' + stdout + ', stderr=' + stderr)
    else:
        # Cache hit!
        logging.debug('cache hit!')


    res = HttpResponse()
    res['Content-Type'] = 'image/png'
    logging.debug('reading cache file ' + fn)
    f = open(fn, 'rb')
    res.write(f.read())
    f.close()

    return res

