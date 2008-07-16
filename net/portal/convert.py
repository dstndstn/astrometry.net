import math
import os
import os.path
import re

from astrometry.net import settings
from astrometry.util import image2pnm
from astrometry.util import fits2fits
from astrometry.net.portal.log import log
from astrometry.util.run_command import run_command
from astrometry.util.filetype import filetype_short

class FileConversionError(Exception):
    errstr = None
    def __init__(self, errstr):
        #super(FileConversionError, self).__init__()
        self.errstr = errstr
    def __str__(self):
        return self.errstr

def run_convert_command(cmd, deleteonfail=None):
    log('Command: ' + cmd)
    (rtn, stdout, stderr) = run_command(cmd)
    if rtn:
        errmsg = 'Command failed: ' + cmd + ': ' + stderr
        log(errmsg + '; rtn val %d' % rtn)
        log('out: ' + stdout);
        log('err: ' + stderr);
        if deleteonfail:
            os.unlink(deleteonfail)
        raise FileConversionError(errmsg)

def run_pnmfile(fn):
    cmd = 'pnmfile %s' % fn
    (filein, fileout) = os.popen2(cmd)
    filein.close()
    out = fileout.read().strip()
    log('pnmfile output: ' + out)
    pat = re.compile(r'P(?P<pnmtype>[BGP])M .*, (?P<width>\d*) by (?P<height>\d*) *maxval (?P<maxval>\d*)')
    match = pat.search(out)
    if not match:
        log('No match.')
        return None
    w = int(match.group('width'))
    h = int(match.group('height'))
    pnmtype = match.group('pnmtype')
    maxval = int(match.group('maxval'))
    log('Type %s, w %i, h %i, maxval %i' % (pnmtype, w, h, maxval))
    return (w, h, pnmtype, maxval)

def is_tarball(fn):
    log('is_tarball: %s' % fn)
    types = filetype_short(fn)
    #log('file type: "%s"' % typeinfo)
    #return any([t.startswith('POSIX tar archive') for t in types])
    for t in types:
        if t.startswith('POSIX tar archive'):
            return True
    return False

def get_objs_in_field(job, df):
    objsfn = convert(job, df, 'objsinfield')
    f = open(objsfn)
    objtxt = f.read()
    objtxt = objtxt.decode('utf_8')
    f.close()
    objs = objtxt.strip()
    if len(objs):
        objs = objs.split('\n')
    else:
        objs = []
    return objs

def annotate_command(job):
    hd = False
    wcs = job.get_tan_wcs()
    if wcs:
        # one square degree
        hd = (wcs.get_field_area() < 1.)
    wcsfn = job.get_filename('wcs.fits')
    cmd = 'plot-constellations -w %s -N -C -B -b 10 -j' % wcsfn
    if hd:
        cmd += ' -D -d %s' % settings.HENRY_DRAPER_CAT
    return cmd

def convert(job, df, fn, args=None):
    if args is None:
        args = {}
    log('convert(%s, args=%s)' % (fn, str(args)))
    tempdir = settings.TEMPDIR
    basename = os.path.join(tempdir, job.get_id() + '-')
    fullfn = basename + fn
    exists = os.path.exists(fullfn)

    variant = 0
    if 'variant' in args:
        variant = int(args['variant'])

    if exists and len(args) == 0:
        return fullfn

    if fn in [ 'uncomp', 'uncomp-js' ]:
        orig = df.get_path()
        comp = image2pnm.uncompress_file(orig, fullfn)
        if comp:
            log('Input file compression: %s' % comp)
        if comp is None:
            return orig
        return fullfn

    elif fn == 'pnm':
        infn = convert(job, df, 'uncomp')
        log('Converting %s to %s...\n' % (infn, fullfn))
        (filetype, errstr) = image2pnm.image2pnm(infn, fullfn, None, False, False, None, False)
        if errstr:
            err = 'Error converting image file: %s' % errstr
            log(err)
            raise FileConversionError(errstr)
        df.filetype = filetype
        return fullfn

    elif fn == 'getimagesize':
        infn = convert(job, df, 'pnm')
        x = run_pnmfile(infn)
        if x is None:
            raise FileConversionError('couldn\'t find file size')
        (w, h, pnmtype, maxval) = x
        log('Type %s, w %i, h %i' % (pnmtype, w, h))
        df.imagew = w
        df.imageh = h
        return None

    elif fn == 'pgm':
        # run 'pnmfile' on the pnm.
        infn = convert(job, df, 'pnm')
        x = run_pnmfile(infn)
        if x is None:
            raise FileConversionError('couldn\'t find file size')
        (w, h, pnmtype, maxval) = x
        log('Type %s, w %i, h %i' % (pnmtype, w, h))
        df.imagew = w
        df.imageh = h
        if pnmtype == 'G':
            return infn
        cmd = 'ppmtopgm %s > %s' % (infn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn in [ 'ppm', 'ppm-8bit' ]:
        eightbit = (fn == 'ppm-8bit')
        if job.is_input_fits() or job.is_input_text():
            # just create the red-circle plot.
            xylist = job.get_axy_filename()
            cmd = ('plotxy -i %s -W %i -H %i -x 1 -y 1 -C brightred -w 2 -P > %s' %
                   (xylist, df.imagew, df.imageh, fullfn))
            run_convert_command(cmd)
            return fullfn

        imgfn = convert(job, df, 'pnm')
        x = run_pnmfile(imgfn)
        if x is None:
            raise FileConversionError('pnmfile failed')
        (w, h, pnmtype, maxval) = x
        if pnmtype == 'P':
            if eightbit and (maxval != 255):
                cmd = 'pnmdepth 255 "%s" > "%s"' % (imgfn, fullfn)
                run_convert_command(cmd)
                return fullfn
            return imgfn
        if eightbit and (maxval != 255):
            cmd = 'pnmdepth 255 "%s" | pgmtoppm white > "%s"' % (imgfn, fullfn)
        else:
            cmd = 'pgmtoppm white "%s" > "%s"' % (imgfn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'fitsimg':
        # check the uncompressed input image type...
        infn = convert(job, df, 'uncomp')
        (df.filetype, cmd, errmsg) = image2pnm.get_image_type(infn)
        if errmsg:
            log(errmsg)
            raise FileConversionError(errmsg)

        # fits image: fits2fits it.
        log('image filetype: ', df.filetype)
        if df.filetype == image2pnm.fitsext:
            errmsg = fits2fits.fits2fits(infn, fullfn, False)
            if errmsg:
                log(errmsg)
                raise FileConversionError(errmsg)
            return fullfn

        # else, convert to pgm and run pnm2fits.
        infn = convert(job, df, 'pgm')
        cmd = 'pnmtofits %s > %s' % (infn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn in [ 'xyls', 'xyls-exists?' ]:
        check_exists = (fn == 'xyls-exists?')
        if variant:
            fn = 'xyls-%i' % variant
            fullfn = basename + fn
            if os.path.exists(fullfn):
                return fullfn

        if check_exists:
            if os.path.exists(fullfn):
                return fullfn
            return None

        if job.is_input_text():
            infn = convert(job, df, 'uncomp')
            cmd = 'xylist2fits %s %s' % (infn, fullfn)
            run_convert_command(cmd)
            return fullfn

        if job.is_input_fits():
            # For FITS bintable uploads: put it through fits2fits.
            infn = convert(job, df, 'uncomp')
            errmsg = fits2fits.fits2fits(infn, fullfn, False)
            if errmsg:
                log(errmsg)
                raise FileConversionError(errmsg)
            return fullfn

        sxylog = 'blind.log'

        extraargs = ''
        if variant:
            altargs = {
                1: '',
                2: '',
                3: '',
                4: '',
                }
            if variant in altargs:
                extraargs = altargs[variant]
            # HACK
            sxylog = '/tmp/alternate-xylist-%s.log' % variant

        infn = convert(job, df, 'fitsimg')
        cmd = 'image2xy -v %s-o %s %s >> %s 2>&1' % (extraargs, fullfn, infn, sxylog)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'xyls-half':
        infn = convert(job, df, 'fitsimg')
        sxylog = 'blind.log'
        cmd = 'image2xy -H -o %s %s >> %s 2>&1' % (fullfn, infn, sxylog)
        run_convert_command(cmd)
        return fullfn

    elif fn in [ 'xyls-sorted', 'xyls-half-sorted' ]:
        if fn == 'xyls-sorted':
            infn = convert(job, df, 'xyls')
        else:
            infn = convert(job, df, 'xyls-half')
        logfn = 'blind.log'
        cmd = 'resort-xylist -d %s %s 2>> %s' % (infn, fullfn, logfn)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'wcsinfo':
        infn = job.get_filename('wcs.fits')
        cmd = 'wcsinfo %s > %s' % (infn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'objsinfield':
        cmd = annotate_command(job)
        cmd += ' -L > %s' % fullfn
        run_convert_command(cmd)
        return fullfn

    elif fn == 'fullsizepng':
        fullfn = job.get_filename('fullsize.png')
        if os.path.exists(fullfn):
            return fullfn
        if job.is_input_fits() or job.is_input_text():
            # just create the red-circle plot.
            xylist = job.get_axy_filename()
            cmd = ('plotxy -i %s -W %i -H %i -x 1 -y 1 -C brightred -w 2 > %s' %
                   (xylist, df.imagew, df.imageh, fullfn))
            run_convert_command(cmd)
            return fullfn
        infn = convert(job, df, 'pnm')
        cmd = 'pnmtopng %s > %s' % (infn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'jpeg-norm':
        infn = convert(job, df, 'pnm')
        #cmd = 'pnmtojpeg %s > %s' % (infn, fullfn)
        #cmd = 'pnmnorm %s | pnmtojpeg > %s' % (infn, fullfn)
        cmd = 'pnmnorm -keephues -wpercent 1 %s | pnmtojpeg > %s' % (infn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn in [ 'indexxyls', 'index.xy.fits' ]:
        wcsfn = job.get_filename('wcs.fits')
        indexrdfn = job.get_filename('index.rd.fits')
        if not (os.path.exists(wcsfn) and os.path.exists(indexrdfn)):
            errmsg('indexxyls: WCS and Index rdls files don\'t exist.')
            raise FileConversionError(errmsg)
        cmd = 'wcs-rd2xy -w %s -i %s -o %s' % (wcsfn, indexrdfn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'field.rd.fits':
        wcsfn = job.get_filename('wcs.fits')
        fieldxyfn = job.get_axy_filename()
        if not (os.path.exists(wcsfn) and os.path.exists(fieldxyfn)):
            errmsg('indexxyls: WCS and Index rdls files don\'t exist.')
            raise FileConversionError(errmsg)
        cmd = 'wcs-xy2rd -w %s -i %s -o %s' % (wcsfn, fieldxyfn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn == 'thumbnail':
        imgfn = convert(job, df, 'pnm-thumb')
        cmd = 'pnmtopng %s > %s' % (imgfn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn in [ 'pnm-small', 'pnm-medium', 'pnm-thumb' ]:
        small = (fn == 'pnm-small')
        thumb = (fn == 'pnm-thumb')
        imgfn = convert(job, df, 'pnm')
        log('in convert(%s): df = %s' % (fn, str(df)))
        if small:
            (scale, w, h) = df.get_small_scale()
        elif thumb:
            (scale, w, h) = df.get_thumbnail_scale()
        else:
            (scale, w, h) = df.get_medium_scale()
        if scale == 1:
            return imgfn
        #cmd = 'pnmscale -reduce %g %s > %s' % (float(scale), imgfn, fullfn)
        cmd = 'pnmscale -width=%g -height=%g %s > %s' % (w, h, imgfn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn in [ 'ppm-small', 'ppm-thumb', 'ppm-medium',
                 'ppm-small-8bit', 'ppm-thumb-8bit', 'ppm-medium-8bit' ]:
        eightbit = fn.endswith('-8bit')
        small = '-small' in fn
        thumb = '-thumb' in fn

        # what the heck is this??
        if job.is_input_fits() or job.is_input_text():
            # just create the red-circle plot.
            if small:
                (scale, dw, dh) = df.get_small_scale()
            else:
                (scale, dw, dh) = df.get_medium_scale()
            if scale == 1:
                if eightbit:
                    return convert(job, df, 'ppm-8bit')
                else:
                    return convert(job, df, 'ppm')
            xylist = job.get_axy_filename()
            cmd = ('plotxy -i %s -W %i -H %i -s %i -x 1 -y 1 -C brightred -w 2 -P > %s' %
                   (xylist, dw, dh, scale, fullfn))
            run_convert_command(cmd)
            return fullfn

        if small:
            imgfn = convert(job, df, 'pnm-small')
        elif thumb:
            imgfn = convert(job, df, 'pnm-thumb')
        else:
            imgfn = convert(job, df, 'pnm-medium')
        x = run_pnmfile(imgfn)
        if x is None:
            raise FileConversionError('pnmfile failed')
        (w, h, pnmtype, maxval) = x
        if pnmtype == 'P':
            if eightbit and (maxval != 255):
                cmd = 'pnmdepth 255 "%s" > "%s"' % (imgfn, fullfn)
                run_convert_command(cmd)
                return fullfn
            return imgfn
        if eightbit and (maxval != 255):
            cmd = 'pnmdepth 255 "%s" | pgmtoppm white > "%s"' % (imgfn, fullfn)
        else:
            cmd = 'pgmtoppm white %s > %s' % (imgfn, fullfn)
        run_convert_command(cmd)
        return fullfn

    elif fn.startswith('annotation'):
        if fn == 'annotation-big':
            imgfn = convert(job, df, 'ppm-8bit')
            scale = 1.0
        elif fn == 'annotation':
            imgfn = convert(job, df, 'ppm-medium-8bit')
            (scale, dw, dh) = df.get_medium_scale()
        elif fn == 'annotation-small':
            imgfn = convert(job, df, 'ppm-small-8bit')
            (scale, dw, dh) = df.get_small_scale()
        elif fn == 'annotation-thumb':
            imgfn = convert(job, df, 'ppm-thumb-8bit')
            (scale, dw, dh) = df.get_thumbnail_scale()


        cmd = annotate_command(job)
        cmd += ' -o %s -s %g -i %s' % (fullfn, 1.0/float(scale), imgfn)
        if 'grid' in args:
            cmd += ' -G %g' % args['grid']
        run_convert_command(cmd)
        return fullfn

    elif fn.startswith('redgreen'):
        # 'redgreen' or 'redgreen-big'
        if 'red' in args:
            red = args['red']
        else:
            red = 'cc3333'

        if 'rmarker' in args:
            rmarker = args['rmarker']
        else:
            rmarker = 'circle'

        if 'green' in args:
            green = args['green']
        else:
            green = '66ccff' # '9999ff' #

        if 'gmarker' in args:
            gmarker = args['gmarker']
        else:
            gmarker = 'Xcrosshair'

        fullfn = basename + fn + '-' + red + '-' + rmarker + '-' + green + '-' + gmarker
        if os.path.exists(fullfn):
            return fullfn

        if fn == 'redgreen':
            imgfn = convert(job, df, 'ppm-medium-8bit')
            (dscale, dw, dh) = df.get_medium_scale()
            scale = 1.0 / float(dscale)
        else:
            imgfn = convert(job, df, 'ppm-8bit')
            scale = 1.0
        fxy = job.get_axy_filename()
        match = job.get_filename('match.fits')
        ixy = convert(job, df, 'index-xy')
        commonargs = ' -S %f -x %f -y %f -w 2' % (scale, scale, scale)
        logfn = 'blind.log'
        if 0:
            cmd = ('plotxy -i %s -I %s -r 6 -C %s -s %s -N 50 -P' % (fxy, imgfn, red, rmarker) + commonargs
                   + '| plotxy -i %s -I - -r 4 -C %s -s %s -n 50 -P' % (fxy, red, rmarker) + commonargs
                   + '| plotxy -i %s -I - -r 4 -C %s -s %s' % (ixy, green, gmarker) + commonargs
                   + ' > %s' % (fullfn))
        cmd = ('plotxy -i %s -I %s -r 5 -C %s -s %s -P' % (fxy, imgfn, red, rmarker) + commonargs
               + '| plotxy -i %s -I - -r 5 -C %s -s %s -b black -P' % (ixy, green, gmarker) + commonargs
               + '| plotquad -m %s -I - -C %s -s %g -w 2 -b black' % (match, green, scale)
               + ' > %s' % (fullfn))
        run_convert_command(cmd, fullfn)
        return fullfn

    elif fn == 'index-xy':
        irdfn = job.get_filename('index.rd.fits')
        wcsfn = job.get_filename('wcs.fits')
        cmd = 'wcs-rd2xy -q -w %s -i %s -o %s' % (wcsfn, irdfn, fullfn)
        run_convert_command(cmd, fullfn)
        return fullfn

    elif fn == 'sources-small':
        imgfn = convert(job, df, 'ppm-small-8bit')
        if variant:
            fn = 'sources-small-%i' % variant
            fullfn = basename + fn
            if os.path.exists(fullfn):
                return fullfn
            xyls = convert(job, df, 'xyls', args)
        else:
            xyls = job.get_axy_filename()
        (dscale, dw, dh) = df.get_small_scale()
        scale = 1.0 / float(dscale)
        commonargs = ('-i %s -x %g -y %g -w 1 -S %g -C red' %
                      (xyls, scale, scale, scale))
        cmd = (('plotxy %s -I %s -N 100 -r 5 -P' %
                (commonargs, imgfn)) +
               (' | plotxy -I - %s -n 100 -N 500 -r 4 > %s' %
                (commonargs, fullfn)))
        log('Command: ' + cmd)
        run_convert_command(cmd, fullfn)
        return fullfn

    elif fn == 'sources-medium':
        imgfn = convert(job, df, 'ppm-medium-8bit')
        xyls = job.get_axy_filename()
        (dscale, dw, dh) = df.get_medium_scale()
        scale = 1.0 / float(dscale)
        commonargs = ('-i %s -x %g -y %g -w 2 -S %g -C red' %
                      (xyls, scale, scale, scale))
        cmd = (('plotxy %s -I %s -N 100 -r 6 -P' %
                (commonargs, imgfn)) +
               (' | plotxy -I - %s -n 100 -N 500 -r 4 > %s' %
                (commonargs, fullfn)))
        run_convert_command(cmd, fullfn)
        return fullfn

    elif fn == 'sources-big':
        imgfn = convert(job, df, 'ppm-8bit')
        xyls = job.get_axy_filename()
        commonargs = ('-i %s -x %g -y %g -w 2 -C red' %
                      (xyls, 1, 1))
        cmd = (('plotxy %s -I %s -N 100 -r 6 -P' %
                (commonargs, imgfn)) +
               (' | plotxy -I - %s -n 100 -N 500 -r 4 > %s' %
                (commonargs, fullfn)))
        log('Command: ' + cmd)
        run_convert_command(cmd, fullfn)
        return fullfn

    elif fn == 'onsky-dot':
        wcsfn = job.get_filename('wcs.fits')
        (ramin,ramax,decmin,decmax) = (0, 360, -85, 85)
        (w, h) = (300, 300)
        lw = 3
        layers = [ 'tycho', 'grid', 'userdot' ]
        gain = -1
        cmd = ('tilerender'
               + ' -x %f -X %f -y %f -Y %f' % (ramin, ramax, decmin, decmax)
               + ' -w %i -h %i' % (w, h)
               + ' -L %g' % lw
               + ''.join((' -l ' + l) for l in layers)
               + ' -I %s' % wcsfn
               + ' -s' # arcsinh
               + ' -g %g' % gain
               + ' > %s' % fullfn
               )
        run_convert_command(cmd, fullfn)
        return fullfn

    errmsg = 'Unimplemented: convert(%s)' % fn
    log(errmsg)
    raise FileConversionError(errmsg)

