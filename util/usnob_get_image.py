#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE


import re
import sys
import socket
import time
from urllib.request import urlopen
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import urlparse, urljoin
from os.path import basename
from optparse import OptionParser

from astrometry.util.starutil_numpy import *
from astrometry.util.file import *

# raw, decw: rectangle size, in arcmin.
# returns list of local filenames to which images were written
def get_usnob_images(ra, dec, raw=15., decw=15., fieldname='',
                     fits=False, wait=True, sources=None, basefn=None,
                     survey='All Surveys', justurls=False):
    if survey is None:
        survey = 'All Surveys'
    if survey == 'poss-i':
        survey = 'POSS-I (103aO, 103aE)'
    if survey == 'poss-ii':
        survey = 'POSS-II (IIIaJ, IIIaF, IV-N)'
    # other survey options include 'SOUTH' and 'AAO-R'

    formvals = {
        'fld': fieldname,
        'ra': ra2hmsstring(ra),
        'dec': dec2dmsstring(dec),
        'rawid': raw,
        'decwid': decw,
        'wunits': 'Minutes',
        'surims': survey,
        'pixflg': 'yes', # images, please
        'whorbl': 'Light Stars/Dark Sky',
        'pixgraph': 'JPEG',
        'cextract': 'rect',
        'ori': 'NW - North Up, East Left',
        'cat': 'USNO B1.0',
        }

    if fits:
        formvals['pixfits'] = 'Yes'
    else:
        formvals['pixfits'] = 'No'
    if sources is None:
        formvals.update({
            'opstars': 'No', # overplot stars?
            'colbits': 'cb_ra',
            })

    else:
        knownsources = ['R1','R2','B1','B2']
        if not sources in knownsources:
            print('Unknown source color', sources)
            print('options:', knownsources)
            return -1
        formvals.update({'opstars': 'Yes',
                         'cat': 'USNO B1.0',
                         'getcat': 'yes',
                         'colbits': 'cb_ra',
                         'clr': 'R1', # overplot color?
                         })

    if basefn is None:
        basefn = 'usnob-%g-%g-' % (ra,dec)

    #queryurl = 'http://www.nofs.navy.mil/cgi-bin/tfch3tI.cgi'
    queryurl = 'http://www.nofs.navy.mil/cgi-bin/tfch4.test.cgi'
    print('submitting form values:')
    for k,v in list(formvals.items()):
        print('  ',k,'=',v)
    print('encoded as:')
    print('  ' + urlencode(formvals))
    print()
    print('waiting for results...')
    socket.setdefaulttimeout(300)
    f = urlopen(queryurl, urlencode(formvals))
    doc = f.read()
    fn = 'res.html'
    write_file(doc, fn)
    m = re.search(r'<A HREF="(.*)">', doc)
    if not m:
        raise RuntimeError('Failed to parse results page: server output written to file ' + fn)
    resurl = m.group(1)
    print('result url', resurl)

    if not wait:
        return resurl
    fns = []
    # keep hitting the url... it may be 404 for a while.
    while True:
        print('requesting result url', resurl)
        res = urlopen(resurl)
        #print 'result info:', res.info()
        print()
        doc = res.read()
        write_file(doc, 'res2.html')

        mjpeg = re.findall(r'<A HREF="([a-zA-z0-9/_.]*?)  ">jpg  </A>', doc)
        print('got jpeg matches:', mjpeg)
        print()
        if not len(mjpeg):
            raise Exception('no jpeg results found')
        if fits:
            mfits = re.findall(r'<A HREF="(.*?)">FITS</A>', doc)
            print('got fits matches:', mfits)
            print()
            if not len(mfits):
                raise Exception('no fits results found')
        else:
            mfits = []

        if justurls:
            if fits:
                return ([urljoin(resurl, url) for url in mjpeg],
                        [urljoin(resurl, url) for url in mfits])
            else:
                return [urljoin(resurl, url) for url in mjpeg]

        for (urls,suffix) in [(mjpeg,''), (mfits,'.fits')]:
            for url in urls:
                print('retrieving image', url)
                fullurl = urljoin(resurl, url)
                print('retrieving url', fullurl)
                fn = basefn + basename(url) + suffix
                for trynum in range(100):
                    try:
                        res = urlopen(fullurl)
                        print()
                        print('writing to file', fn)
                        write_file(res.read(), fn)
                        break
                    except Exception as e:
                        print('Error:', e)
                        print('Retrying...')
                        time.sleep(1)
                fns.append(fn)
        break
    return fns


if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <ra> <dec>')

    parser.add_option('--r1', '--R1', dest='red1', help='overplot first-epoch red sources', action='store_true')
    parser.add_option('--r2', '--R2', dest='red2', help='overplot second-epoch red sources', action='store_true')
    parser.add_option('-b', dest='big', help='30x30 arcmin (default: 15x15)', action='store_true')
    parser.add_option('-f', '--fits', dest='fits', help='Get FITS image too', action='store_true')
    parser.add_option('-n', dest='fieldname', help='fieldname (default: NoName)')

    parser.set_defaults(red1=False, red2=False, big=None, fieldname=None, fits=False)

    (opt, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        print()
        print('Got extra arguments:', args)
        sys.exit(-1)

    sources = None
    if opt.red1:
        sources = 'R1'
    elif opt.red2:
        sources = 'R2'

    sz = 15.
    if opt.big:
        sz = 30.

    # parse RA,Dec.
    ra = float(args[0])
    dec = float(args[1])

    args = {}
    if opt.fieldname:
        args['fieldname'] = opt.fieldname
    if opt.fits:
        args['fits'] = True
    get_usnob_images(ra, dec, sources=sources, raw=sz, decw=sz, **args)
