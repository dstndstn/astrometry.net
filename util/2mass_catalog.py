#! /usr/bin/env python3
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

def get_2mass_sources(ra, dec, radius=1, basefn=None):
    twomass_nside = 9

    if basefn is None:
        twomass_pat = '2mass_hp%03i.fits'
    else:
        twomass_pat = basefn

    hps = healpix_rangesearch(ra, dec, radius, twomass_nside)
    print('2MASS healpixes in range:', hps)
    allU = None
    for hp in hps:
        fn = twomass_pat % hp
        print('2MASS filename:', fn)
        U = fits_table(fn)
        print(len(U), 'sources')
        I = (degrees_between(ra, dec, U.ra, U.dec) < radius)
        print('%i 2MASS stars within range.' % sum(I))
        U = U[I]
        if allU is None:
            allU = U
        else:
            allU.append(U)
    return allU

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <ra> <dec> <output-filename>')

    parser.add_option('-r', dest='radius', type='float', help='Search radius, in deg (default 1 deg)')
    parser.add_option('-b', dest='basefn', help='Base filename of 2MASS FITS files (default: 2mass_hp%03i.fits)')
    parser.add_option('-B', dest='band', help='Band (J, H, or K) to use for cuts')
    parser.set_defaults(radius=1.0, basefn=None, band='J')

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

    band = opt.band.lower()

    # ugh!
    opts = {}
    for k in ['radius', 'basefn']:
        opts[k] = getattr(opt, k)

    X = get_2mass_sources(ra, dec, **opts)
    print('Got %i 2MASS sources.' % len(X))

    #print X.about()

    print('Applying cuts...')
    I = logical_not(X.minor_planet)
    print('not minor planet:', sum(I))
    qual = X.get(band + '_quality')
    # work around dumb bug where it's a single-char column rather than a byte.
    nobrightness = chr(0)
    I = logical_and(I, (qual != nobrightness))
    print('not NO_BRIGHTNESS', sum(I))
    print(len(X))
    print(len(X.j_cc))
    cc = array(X.getcolumn(band + '_cc'))
    ccnone = chr(0)
    #print 'cc shape', cc.shape
    #print cc[:10]
    #print ccnone
    I = logical_and(I, (cc == ccnone))
    print('CC_NONE', sum(I))
    X = X[I]
    print('%i pass cuts' % len(X))

    print('Writing to', outfn)
    X.write_to(outfn)
    
