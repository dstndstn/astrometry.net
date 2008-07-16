from django.http import *
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse
from django.db.models import Q

from astrometry.net.tile.models import Image
from astrometry.net.portal.job import Submission, Job, DiskFile
from astrometry.net.portal.convert import convert

import re
import os.path
import os
import popen2
# grab a stash of hash
#import hashlib
# old hash
# md5 was causing Apache segfaults for me...
#import md5
import sha
import logging
import commands
import shutil

import astrometry.util.sip

import ctypes

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

def index(request):
    ctxt = {
        'gmaps_key' : ('http://maps.google.com/maps?file=api&v=2.x&key=' +
                       settings.GMAPS_API_KEY),
        'map_js_url' : reverse('astrometry.net.media') + 'map.js',
        'gmaps_tile_url' : reverse(get_tile) + '?',
        'gmaps_image_url' : reverse(get_image)+ '?',
        'gmaps_image_list_url' : reverse(get_image_list) + '?',
        'gmaps_black_url' : reverse('astrometry.net.media') + 'black.png',
        }
    t = loader.get_template('tile/index.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

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
    # Flip RA!
    (ramin,  ramax ) = (-longmax, -longmin)
    (decmin, decmax) = ( latmin,   latmax )
    # The Google Maps client treats RA as going from -180 to +180; we prefer
    # to think of it as going from 0 to 360.  If the low value is negative,
    # wrap it around...
    if (ramin < 0.0):
        ramin += 360
        ramax += 360
    return (ramin, ramax, decmin, decmax)

def get_image(request):
    logging.debug("get_image() starting")
    try:
        fn = request.GET['filename']
    except KeyError:
        return HttpResponse('No filename specified.')
    if not filename_ok(fn):
        return HttpResponse('Bad filename')
    q = list(Image.objects.filter(filename=fn))
    if not len(q):
        return HttpResponse('No such file.')
    img = q[0]

    # Content-type
    ctmap = { 'jpeg':'image/jpeg' }

    res = HttpResponse()
    res['Content-Type'] = ctmap[img.origformat]
    path = settings.imgdir + "/" + img.origfilename
    logging.debug("Opening file " + path)
    f = open(path, "rb")
    res.write(f.read())
    f.close()
    return res

def get_overlapping_images(ramin, ramax, decmin, decmax):
    dec_ok = Image.objects.filter(decmin__lte=decmax, decmax__gte=decmin)
    Q_normal = Q(ramin__lte=ramax) & Q(ramax__gte=ramin)
    # In the database, any image that spans the RA=0 line has its bounds
    # bumped up by 360; therefore every "ramin" value is > 0, but some
    # "ramax" values are > 360.
    raminwrap = ramin + 360
    ramaxwrap = ramax + 360
    Q_wrap   = Q(ramin__lte=ramaxwrap) & Q(ramax__gte=raminwrap)
    inbounds = dec_ok.filter(Q_normal | Q_wrap)
    return inbounds

def filename_ok(fn):
    if fn.find('..') != -1:
        return False
    # valid filenames regexp
    filenameRE = re.compile(r'^[A-Za-z0-9\./_-]+$')
    if filenameRE.match(fn) is None:
        return False
    return True

def get_image_list(request):
    logging.debug("imagelist() starting")
    try:
        (ramin, ramax, decmin, decmax) = getbb(request)
    except KeyError, x:
        return HttpResponse(x)
    logging.debug("Bounds: RA [%g, %g], Dec [%g, %g]." % (ramin, ramax, decmin, decmax))
    inbounds = get_overlapping_images(ramin, ramax, decmin, decmax)

    # We have a query that isolates images that overlap.  We now want to order them by
    # the proportion of overlap:
    #     (overlapping area)^2
    #     --------------------
    #        area_1 * area_2
    # Note that the area of our query rectangle is constant, so we can omit it from
    # the score function.
    #
    # We'll just treat the axis-aligned bounding box as the image, and we'll treat
    # RA and Dec as rectangular units.  Obviously these are wrong, but this is SQL
    # so we can't get too fancy...
    #
    # The overlapping area is just the min of the maxes minus the max of the mins,
    # in RA and Dec.
    # 
    # What about wrap-around images?  Gah!  I guess the overlapping RA is the max of the
    # overlapping normal RA and wrap-around RA (because wrongly wrapped-around RA range
    # becomes negative)

    overlap = ('(max(min(ramax, %g) - max(ramin, %g), min(ramax, %g) - max(ramin, %g)) * (min(decmax, %g) - max(decmin, %g)))' %
               (ramax, ramin, ramax+360, ramin+360, decmax, decmin))
    a1 = '((ramax - ramin) * (decmax - decmin))'
    #a2 = str((ramax - ramin) * (decmax - decmin))
    sortbysize = inbounds.order_by_expression('-(' + overlap + '*' + overlap + ') / ' + a1)

    top20 = sortbysize[:20]
    query = top20

    res = HttpResponse()
    res['Content-Type'] = 'text/xml'
    res.write('<imagelist>\n')

    a2 = (ramax - ramin) * (decmax - decmin)
    for img in query:
        dra1 = min(ramax, img.ramax) - max(ramin, img.ramin)
        dra2 = min(ramax+360, img.ramax) - max(ramin+360, img.ramin)
        overlap = max(dra1, dra2) * (min(decmax, img.decmax) - min(decmin, img.decmin))
        a1 = ((img.ramax - img.ramin) * (img.decmax - img.decmin))
        score = (overlap**2) / (a1 * a2)
        logging.debug("Image " + img.filename + ": score %g (dra1=%g, dra2=%g))" %
                      (score, dra1, dra2))

        wcsfn = settings.imgdir + '/' + img.filename + '.wcs'
        try:
            if not sip.libraryloaded():
                fn = settings.sipso
                logging.debug('Trying to load library %s' % fn)
                #sip.loadlibrary(fn)
                lib = ctypes.CDLL(fn)
                logging.debug('Lib is ' + str(lib))
                sip._sip = lib

            wcs = sip.Sip(filename=wcsfn)
            poly = []
            steps = 4
            # bottom
            y = 1
            for i in xrange(steps):
                x = 1 + float(i) / (steps-1) * (wcs.wcstan.imagew-1)
                ra,dec = wcs.pixelxy2radec(x, y)
                poly.append(360-ra)
                poly.append(dec)
            # right
            x = wcs.wcstan.imagew
            for i in xrange(steps):
                y = 1 + float(i) / (steps-1) * (wcs.wcstan.imageh-1)
                ra,dec = wcs.pixelxy2radec(x, y)
                poly.append(360-ra)
                poly.append(dec)
            # top
            y = wcs.wcstan.imageh
            for i in xrange(steps-1, -1, -1):
                x = 1 + float(i) / (steps-1) * (wcs.wcstan.imagew-1)
                ra,dec = wcs.pixelxy2radec(x, y)
                poly.append(360-ra)
                poly.append(dec)
            # left
            x = 0
            for i in xrange(steps-1, -1, -1):
                y = 1 + float(i) / (steps-1) * (wcs.wcstan.imageh-1)
                ra,dec = wcs.pixelxy2radec(x, y)
                poly.append(360-ra)
                poly.append(dec)

            poly = ','.join(map(str, poly))
        except Exception, e:
            logging.debug('Failed to read SIP header from %s: %s' % (wcsfn, e))
            poly=''

        #latmin = img.decmin
        #latmax = img.decmax
        #longmin = 360-img.ramax
        #longmax = 360-img.ramin
        #poly = map(str, (longmin, latmin, longmin, latmax, longmax, latmax, longmax, latmin, longmin, latmin))
        #if (score < 0.01**2):
        #   continue
        #res.write('<image name="%s" poly="%s" />\n' % (img.filename, ','.join(poly)))
        res.write('<image name="%s"' % img.filename)
        if len(poly):
            res.write(' poly="%s"' % poly)
        res.write(' />')

    res.write('</imagelist>\n')
    logging.debug("Returning %i files." % len(query))
    return res

def get_tile(request):
    #logging.debug('query() starting')
    try:
        (ramin, ramax, decmin, decmax) = getbb(request)
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

    # Build tilerender command-line.
    # RA,Dec range; image size.
    cmdline = tilerender + (" -x %f -X %f -y %f -Y %f" % (ramin, ramax, decmin, decmax))
    cmdline += (" -w %i -h %i" % (imw, imh))

    if 'toright' in request.GET:
        cmdline += ' -z'

    if 'lw' in request.GET:
        lw = float(request.GET['lw'])
        if lw:
            cmdline += ' -L %g' % lw

    # cachedir: -D
    cmdline += (" -D " + rendercachedir)
    # layers: -l
    layerexp = re.compile(r'^\w+$')
    #cmdline += (" -l %s" % lay) for lay in layers if layerexp.match(lay)
    for lay in layers:
        if layerexp.match(lay):
            cmdline += (" -l " + lay)

    if ('ubstyle' in request.GET):
        style = request.GET['ubstyle']
        cmdline += (' -b ' + style)

    if ('imagefn' in request.GET):
        imgfns = request.GET['imagefn'].split(',')
        for img in imgfns:
            if not filename_ok(img):
                logging.debug("Bad image filename: \"" + img + "\".")
                return HttpResponse('bad filename.')
            cmdline += (" -i " + img)

    if ('wcsfn' in request.GET):
        wcsfns = request.GET['wcsfn'].split(',')
        for wcs in wcsfns:
            if not filename_ok(wcs):
                logging.debug("Bad WCS filename: \"" + wcs + "\".")
                return HttpResponse('bad filename.')
            cmdline += (" -I " + wcs)

    if ('rdlsfn' in request.GET):
        rdlsfns = request.GET['rdlsfn'].split(',')
        for rdls in rdlsfns:
            if not filename_ok(rdls):
                logging.debug("Bad RDLS filename: \"" + rdls + "\".");
                return HttpResponse('bad filename.')
            cmdline += (' -r ' + rdls)

    if ('images' in layers) or ('boundaries' in layers):
        # filelist: -S

        filenames = []

        if 'submission' in request.GET:
            subid = request.GET['submission']
            subs = Submission.objects.all().filter(subid=subid)
            if len(subs) != 1:
                logging.debug("Found %i submission matching id %s, expected 1.\n" % (len(subs), subid))
                return HttpResponse('Got %i submissions.' % len(subs))
            sub = subs[0]
            jobs = sub.jobs.all().filter(status='Solved')

            # BIG HACK!
            if not sip.libraryloaded():
                sip.loadlibrary('/home/gmaps/test/an-common/_sip.so')

            for job in jobs:
                fn = tempdir + '/' + 'gmaps-' + job.jobid
                wcsfn = fn + '.wcs'
                jpegfn = fn + '.jpg'

                if not os.path.exists(wcsfn):
                    calib = job.calibration
                    tanwcs = calib.raw_tan
                    if not tanwcs:
                        continue
                    # convert to an_common.sip.Tan
                    tanwcs = tanwcs.to_tanwcs()
                    # write to .wcs file.
                    tanwcs.write_to_file(wcsfn)
                    logging.debug('Writing WCS file ' + wcsfn)

                if not os.path.exists(jpegfn):
                    logging.debug('Writing JPEG file ' + jpegfn)
                    tmpjpeg = convert(job, 'jpeg-norm')
                    shutil.copy(tmpjpeg, jpegfn)

                filenames.append(fn)

        else:
            # Compute list of files via DB query
            imgs = get_overlapping_images(ramin, ramax, decmin, decmax)
            # Get list of filenames
            filenames = [img.filename for img in imgs]

        files = "\n".join(filenames) + "\n"
        logging.debug("For RA in [%f, %f] and Dec in [%f, %f], found %i files." %
                      (ramin, ramax, decmin, decmax, len(filenames)))

        # Compute filename
        m = sha.new()
        m.update(files)
        digest = m.hexdigest()
        fn = tempdir + '/' + digest
        # Write to that filename
        try:
            f = open(fn, 'wb')
            f.write(files)
            f.close()
        except (IOError):
            return HttpResponse('Failed to write file list.')
        cmdline += (" -S " + fn)

    # Options with no args:
    optflags = { 'jpeg'   : '-J',
                 'arcsinh': '-s',
                 }
    for opt,arg in optflags.iteritems():
        if (opt in request.GET):
            cmdline += (' ' + arg)

    # Options with numeric args.
    optnum = { 'dashbox' : '-B',
               'gain'    : '-g',
               }
    for opt,arg in optnum.iteritems():
        if (opt in request.GET):
            num = float(request.GET[opt])
            cmdline += (" %s %f" % (arg, num))

    # Options with choice args.
    optchoice = { 'colormap' : {'arg':'-C', 'valid':['rb', 'i']},
                  }

    for opt,choice in optchoice.iteritems():
        if (opt in request.GET):
            val = request.GET[opt]
            valid = choice['valid']
            if (val in valid):
                arg = choice['arg']
                cmdline += (' ' + arg + ' ' + val)

    jpeg = ('jpeg' in request.GET)

    res = HttpResponse()
    if jpeg:
        res['Content-Type'] = 'image/jpeg'
    else:
        res['Content-Type'] = 'image/png'
    #logging.debug('command-line is ' + cmdline)

    if ('tag' in request.GET):
        tag = request.GET['tag']
        if not re.match('^\w+$', tag):
            return HttpResponse('Bad tag')
        tilecachedir = cachedir + '/' + tag
        if not os.path.exists(tilecachedir):
            os.mkdir(tilecachedir)

        # Compute filename
        #m = hashlib.md5()
        m = sha.new()
        m.update(cmdline)
        fn = tilecachedir + '/' + 'tile-' + m.hexdigest() + '.'
        if jpeg:
            fn += 'jpg'
        else:
            fn += 'png'

        logging.debug('tilecache file: ' + fn)
        if not os.path.exists(fn):
            # Run it!
            cmd = cmdline + ' > ' + fn + ' 2>> ' + logfile
            logging.debug('running: ' + cmd)
            rtn = os.system(cmd)
            if not(os.WIFEXITED(rtn) and (os.WEXITSTATUS(rtn) == 0)):
                if (os.WIFEXITED(rtn)):
                    logging.debug('exited with status %d' % os.WEXITSTATUS(rtn))
                else:
                    logging.debug('command did not exit.')
                try:
                    os.remove(fn)
                except (OSError):
                    pass
                return HttpResponse('tilerender command failed.')
        else:
            # Cache hit!
            logging.debug('cache hit!')
            pass

        logging.debug('reading cache file ' + fn)
        f = open(fn, 'rb')
        res.write(f.read())
        f.close()
    else:
        cmd = cmdline + ' 2>> ' + logfile
        logging.debug('no caching: just running command ' + cmd)
        (rtn, out) = commands.getstatusoutput(cmd)
        if not(os.WIFEXITED(rtn) and (os.WEXITSTATUS(rtn) == 0)):
            if (os.WIFEXITED(rtn)):
                logging.debug('exited with status %d' % os.WEXITSTATUS(rtn))
            else:
                logging.debug('command did not exit.')
            return HttpResponse('tilerender command failed.')
        res.write(out)

    logging.debug('finished.')
    return res
