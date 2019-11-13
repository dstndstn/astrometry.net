#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import sys
from optparse import OptionParser

try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
from numpy import *

from astrometry.util.fits import *
from astrometry.util.healpix import *
from astrometry.util.starutil_numpy import *
from astrometry.util.usnob_cuts import *

def get_usnob_sources(ra, dec, radius=1, basefn=None):
    usnob_nside = 9

    if basefn is None:
        usnob_pat = 'usnob10_hp%03i.fits'
    else:
        usnob_pat = basefn

    usnobhps = healpix_rangesearch(ra, dec, radius, usnob_nside)
    print('USNO-B healpixes in range:', usnobhps)
    allU = None
    for hp in usnobhps:
        usnobfn = usnob_pat % hp
        print('USNOB filename:', usnobfn)
        U = table_fields(usnobfn)
        I = (degrees_between(ra, dec, U.ra, U.dec) < radius)
        print('%i USNOB stars within range.' % sum(I))
        U = U[I]
        if allU is None:
            allU = U
        else:
            allU.append(U)
    return allU

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <ra> <dec> <output-filename>')

    parser.add_option('-r', dest='radius', type='float', help='Search radius, in deg (default 1 deg)')
    parser.add_option('-b', dest='basefn', help='Base filename of USNO-B FITS files (default: usnob10_hp%03i.fits)')
    parser.set_defaults(radius=1.0, basefn=None)

    (opt, args) = parser.parse_args()
    if len(args) != 3:
        parser.print_help()
        print()
        print('Got extra arguments:', args)
        sys.exit(-1)

    # parse RA,Dec.
    ra = float(args[0])
    dec = float(args[1])
    outfn = args[2]

    # ugh!
    opts = {}
    for k in ['radius', 'basefn']:
        opts[k] = getattr(opt, k)

    X = get_usnob_sources(ra, dec, **opts)
    print('Got %i USNO-B sources.' % len(X))

    print('Applying cuts...')
    I = usnob_apply_cuts(X)
    X = X[I]
    print(len(X), 'pass cuts')

    usnob_compute_average_mags(X)

    print('Writing to', outfn)
    X.write_to(outfn)
    
