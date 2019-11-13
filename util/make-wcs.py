#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import sys
from optparse import OptionParser
from astrometry.util.util import *

if __name__ == '__main__':
    parser = OptionParser('usage: %prog [options] <outfn>')
    parser.add_option('-r', '--ra', dest='ra', type='float', help='RA (deg)')
    parser.add_option('-d', '--dec', dest='dec', type='float', help='Dec (deg)')
    parser.add_option('-s', '--size', dest='size', type='float', help='Field width (deg)')
    parser.add_option('-p', '--pixscale', type='float', help='Pixel scale (arcsec/pixel)')
    parser.add_option('-W', '--width', dest='w', type='int', help='Image width, default %default', default=1024)
    parser.add_option('-H', '--height', dest='h', type='int', help='Image height, default %default', default=1024)
    parser.set_defaults(ra=None, dec=None, size=None, w=None, h=None)
    opt,args = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(0)

    if opt.pixscale is None and opt.size is None:
        print('Must specify --pixscale or --size')
        parser.print_help()
        sys.exit(-1)
    pixscale = 0.
    if opt.size is not None:
        pixscale = opt.size / opt.w
    elif opt.pixscale is not None:
        pixscale = opt.pixscale / 3600.

    if opt.ra is None or opt.dec is None:
        print('Must specify --ra and --dec')
        parser.print_help()
        sys.exit(-1)

    wcs = Tan(*[float(x) for x in [
        opt.ra, opt.dec, 0.5 + (opt.w / 2.), 0.5 + (opt.h / 2.),
        -pixscale, 0., 0., pixscale, opt.w, opt.h,
    ]])

    wcs.write_to(args[0])
