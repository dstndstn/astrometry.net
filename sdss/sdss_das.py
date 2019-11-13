#! /usr/bin/env python3
from __future__ import print_function

from astrometry.util.run_command import run_command
from astrometry.util.sdss_filenames import *
from astrometry.sdss import *

def get_urls(urls, outfn, curl=False):
    for url in urls:
        if curl:
            cmd = 'curl '
            if outfn:
                cmd += '-o %s ' % outfn
            else:
                cmd += '-O '
        else:
            cmd = 'wget --continue -nv '
            if outfn:
                cmd += '-O %s ' % outfn

        cmd += '\"%s\"' % url
        print('Running:', cmd)
        (rtn, out, err) = run_command(cmd)
        if rtn == 0:
            return True
        if rtn:
            print('Command failed: command', cmd)
            print('Output:', out)
            print('Error:', err)
            print('Return val:', rtn)
    return False

def sdss_das_get_suffix(filetype):
    return ({'fpC': '.gz'}).get(filetype, '')

def sdss_das_get_url(filetype, run, camcol, field, rerun, band, suffix=None):
    if suffix is None:
        suffix = sdss_das_get_suffix(filetype)
    path = sdss_path(filetype, run, camcol, field, band, rerun)
    if path is None:
        print('Unknown SDSS filetype', filetype)
        return None
    return 'http://das.sdss.org/imaging/' + path + suffix

def sdss_das_get(filetype, outfn, run, camcol, field, band=None, reruns=None, suffix=None, gunzip=True, curl=False, ):
    if reruns is None:
        reruns = [40,41,42,44]
    urls = []
    for rerun in reruns:
        url = sdss_das_get_url(filetype, run, camcol, field, rerun, band,
                               suffix=suffix)
        if url is None:
            return False
        urls.append(url)
        
    if suffix is None:
        suffix = sdss_das_get_suffix(filetype)
    if outfn:
        outfn = outfn % { 'run':run, 'camcol':camcol, 'field':field, 'band':band } + suffix
    else:
        outfn = sdss_filename(filetype, run, camcol, field, band) + suffix
    if not get_urls(urls, outfn, curl):
        return False

    if suffix == '.gz' and gunzip:
        print('gzipped file; outfn=', outfn)
        gzipfn = outfn
        outfn = gzipfn.replace('.gz', '')
        if os.path.exists(gzipfn):
            cmd = 'gunzip -cd %s > %s' % (gzipfn, outfn)
            print('Running:', cmd)
            (rtn, out, err) = run_command(cmd)
            if rtn:
                print('Command failed: command', cmd)
                print('Output:', out)
                print('Error:', err)
                print('Return val:', rtn)
    return outfn

def sdss_das_get_fpc(run, camcol, field, band, outfn=None, reruns=None):
    return sdss_das_get('fpC', outfn, run, camcol, field, band, reruns, suffix='.gz')

def sdss_das_get_mask(run, camcol, field, band, outfn=None, reruns=None):
    return sdss_das_get('fpM', outfn, run, camcol, field, band, reruns)


if __name__ == '__main__':
    from optparse import OptionParser
    import sys

    parser = OptionParser(usage=('%prog <options> <file types>\n\n' +
                                 'file types include: fpC, fpM, fpObjc, psField, tsObj, tsField\n'
                                 'and for DR8 and above: frame'))

    parser.add_option('-r', '--run', dest='run', type='int')
    parser.add_option('-f', '--field', dest='field', type='int')
    parser.add_option('-c', '--camcol', dest='camcol', type='int')
    parser.add_option('-R', '--rerun', dest='rerun', type='int')
    parser.add_option('-b', '--band', dest='band')
    parser.add_option('-C', '--curl', dest='curl', action='store_true', default=False)
    parser.add_option('--dr9', dest='dr9', action='store_true',
                      help='Grab DR9 data')
    
    parser.set_defaults(run=None, field=None, camcol=None, band=None, rerun=None)
    (opt, args) = parser.parse_args()
    if not len(args):
        parser.print_help()
        print()
        print('Must specify types of data desired')
        sys.exit(-1)

    run = opt.run
    field = opt.field
    camcol = opt.camcol
    band = opt.band

    if run is None or field is None or camcol is None:
        parser.print_help()
        print()
        print('Must supply --run, --field, --camcol')
        sys.exit(-1)

    argdict = {}
    if band is not None:
        argdict['band'] = band
    if opt.rerun is not None:
        argdict['reruns'] = [opt.rerun]

    if opt.dr9:
        sdss = DR9()
    else:
        sdss = DR7()
        
    for filetype in args:
        print('Retrieving', filetype, '...')
        fn = sdss.retrieve(filetype, run, camcol, field, **argdict)
        #sdss_das_get(filetype, None, run, camcol, field, curl=opt.curl, **argdict)

