import sha
import os.path

from django.http import *
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse

from proj.utils import get_bb, get_imagesize
from proj.run_command import run_command

def get_tile(request):
    try:
        (xmin, xmax, ymin, ymax) = get_bb(request)
        (imw, imh) = get_imagesize(request)
        sx = None
        sy = None
        if 'seedx' in request.GET:
            sx = float(request.GET['seedx'])
        if 'seedy' in request.GET:
            sy = float(request.GET['seedy'])
    except Exception, x:
        return HttpResponse(str(x))

    newymax = 1. - ymin
    ymin = 1. - ymax
    ymax = newymax

    cmdline = ('julia -W %i -H %i -x %g -X %g -y %g -Y %g' %
               (imw, imh, xmin, xmax, ymin, ymax))
    if sx is not None:
        cmdline += ' -s %g' % sx
    if sy is not None:
        cmdline += ' -S %g' % sy

    cmdline += ' | pnmtopng'


    cache = True

    if not cache:
        (rtn, stdout, stderr) = run_command(cmdline)
        if rtn:
            return HttpResponse('Command failed: %s, stdout %s, stderr %s' %
                                (cmdline, stdout, stderr))

        res = HttpResponse()
        res['Content-Type'] = 'image/png'
        res.write(stdout)
        return res

    tempdir = '/tmp'

    m = sha.new()
    m.update(cmdline)
    digest = m.hexdigest()
    fn = tempdir + '/easy-gmaps-' + digest + '.png'

    if not os.path.exists(fn):
        # Run it!
        cmdline += ' > ' + fn
        (rtn, stdout, stderr) = run_command(cmdline)
        if rtn:
            return HttpResponse('Command failed: %s, stdout %s, stderr %s' %
                                (cmdline, stdout, stderr))
    else:
        # Cache hit!
        pass

    res = HttpResponse()
    res['Content-Type'] = 'image/png'
    f = open(fn, 'rb')
    res.write(f.read())
    f.close()
    return res

