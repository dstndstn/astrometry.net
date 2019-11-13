#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from urllib2 import urlopen
from urllib import urlencode
from urlparse import urlparse, urljoin

import os.path

from numpy import *

from astrometry.util.file import *
from astrometry.util.usnob_get_image import *

from optparse import OptionParser


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-r', '--ra-low', '--ra-lo', '--ra-min',
                      dest='ralo', type=float, help='Minimum RA')
    parser.add_option('-R', '--ra-high', '--ra-hi', '--ra-max',
                      dest='rahi', type=float, help='Maximum RA')
    parser.add_option('-d', '--dec-low', '--dec-lo', '--dec-min',
                      dest='declo', type=float, help='Minimum Dec')
    parser.add_option('-D', '--dec-high', '--dec-hi', '--dec-max',
                      dest='dechi', type=float, help='Maximum Dec')
    parser.add_option('-p', '--prefix',
                      dest='prefix', help='Output file prefix')
    parser.add_option('-s', '--survey',
                      dest='survey', help='Grab only one USNOB survey: poss-i, poss-ii, ... (see http://www.nofs.navy.mil/data/fchpix/cfch.html')
    parser.add_option('-P', '--plate',
                      dest='plate', help='Grab only one USNOB plate: "se0161", for example')
    parser.add_option('-c', '--continue',
                      dest='cont', action='store_true', help='Continue a previously interrupted transfer')

    parser.set_defaults(prefix='usnob', survey=None, plate=None,
                        ralo=None, rahi=None, declo=None, dechi=None, cont=False)
    (opt, args) = parser.parse_args()

    if opt.ralo is None or opt.rahi is None or opt.declo is None or opt.dechi is None:
        parser.print_help()
        parser.error('RA,Dec lo,hi are required.')

    radecs = []

    decstep = 14./60.
    Dec = arange(opt.declo, opt.dechi+decstep, decstep)
    for dec in Dec:
        rastep = 14./60./cos(deg2rad(dec))
        RA  = arange(opt.ralo , opt.rahi +rastep , rastep)
        for ra in RA:
            radecs.append((ra,dec))
    radecs = array(radecs)

    # Retrieve them in order of distance from the center of the region...
    #dists = [distsq_between_radecs(r,d, (opt.ralo+opt.rahi)/2., (opt.declo+opt.dechi)/2.)
    #         for (r,d) in radecs]
    dists = distsq_between_radecs(radecs[:,0], radecs[:,1],
                                  (opt.ralo+opt.rahi)/2., (opt.declo+opt.dechi)/2.)
    order = argsort(dists)

    for (ra,dec) in radecs[order]:
        (jpeg,fits) = get_usnob_images(ra, dec, fits=True, survey=opt.survey, justurls=True)
        print('got jpeg urls:', jpeg)
        print('got fits urls:', fits)
        if opt.plate is None:
            keepjpeg = jpeg
            keepfits = fits
        else:
            keepjpeg = [u for u in jpeg if opt.plate in u]
            keepfits = [u for u in fits if opt.plate in u]
            print('keep jpeg urls:', keepjpeg)
            print('keep fits urls:', keepfits)
        base = opt.prefix + '-%.3f-%.3f-' % (ra,dec)
        for url in keepjpeg:
            # like "fchlwFxSl_so0194.000.jpg"
            urlfn = url.split('/')[-1]
            urlfn = urlfn.split('_')[-1]
            fn = base + urlfn
            if opt.cont and os.path.exists(fn):
                print('File', fn, 'exists.')
                continue
            print('retrieving', url, 'to', fn)
            res = urlopen(url)
            write_file(res.read(), fn)
        for url in keepfits:
            urlfn = url.split('/')[-1]
            urlfn = urlfn.split('_')[-1]
            fn = base + urlfn + '.fits'
            if opt.cont and os.path.exists(fn):
                print('File', fn, 'exists.')
                continue
            print('retrieving', url, 'to', fn)
            res = urlopen(url)
            write_file(res.read(), fn)
        
