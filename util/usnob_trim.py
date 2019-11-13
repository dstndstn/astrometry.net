#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

# Used to trim down the "hpslit"-merged USNO-B files before
# building indices out of them.
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


def trim(infn, outfn):
    print('Reading', infn)
    X = fits_table(infn, columns=[
        'num_detections', 'flags', 'an_diffraction_spike',
        'field_1', 'field_3', 'magnitude_1', 'magnitude_3',
        'field_0', 'field_2', 'magnitude_0', 'magnitude_2',
        'ra', 'dec',
        ])
    print('Read', len(X), 'sources')
    print('Applying cuts')
    I = usnob_apply_cuts(X)

    # drop now-unwanted columns
    for c in ['flags', 'an_diffraction_spike',
              'num_detections' ]:
        X.delete_column(c)
    X.cut(I)
    print('Kept', len(X), 'sources')
    del I

    print('Computing average mags')

    X.field_0 = X.field_0.astype(np.int16)
    X.field_1 = X.field_1.astype(np.int16)
    X.field_2 = X.field_2.astype(np.int16)
    X.field_3 = X.field_3.astype(np.int16)
    X.magnitude_0 = X.magnitude_0.astype(np.float32)
    X.magnitude_1 = X.magnitude_1.astype(np.float32)
    X.magnitude_2 = X.magnitude_2.astype(np.float32)
    X.magnitude_3 = X.magnitude_3.astype(np.float32)

    usnob_compute_average_mags(X)
    for c in [
        'field_1', 'field_3', 'magnitude_1', 'magnitude_3',
        'field_0', 'field_2', 'magnitude_0', 'magnitude_2']:
        X.delete_column(c)

    X.r_mag = X.r_mag.astype(np.float32)
    X.b_mag = X.b_mag.astype(np.float32)
    print('Writing output to', outfn)
    X.writeto(outfn)
    del X

if __name__ == '__main__':
    #for hp in range(12):

    if False:
        # fitscopy usnob-07.fits"[#row<100000000]" usnob-07-a.fits
        # fitscopy usnob-07.fits"[#row>=100000000]" usnob-07-b.fits
        infn = 'usnob-07-a.fits'
        outfn = 'usnob-trimmed-07-a.fits'
        trim(infn, outfn)
    if False:
        infn = 'usnob-07-b.fits'
        outfn = 'usnob-trimmed-07-b.fits'
        trim(infn, outfn)
        # cp usnob-trimmed-07-a.fits 07a.fits
        # tabmerge usnob-trimmed-07-b.fits+1 07a.fits+1
        # mv 07a.fits usnob-trimmed-07.fits
        
    if False:
        infn = 'usnob-10-a.fits'
        outfn = 'usnob-trimmed-10-a.fits'
        trim(infn, outfn)
    if True:
        infn = 'usnob-10-b.fits'
        outfn = 'usnob-trimmed-10-b.fits'
        trim(infn, outfn)

    #for hp in range(7,12):
    #for hp in range(8,12):
    for hp in range(11,12):
        infn = 'usnob-%02i.fits' % hp
        outfn = 'usnob-trimmed-%02i.fits' % hp
        trim(infn, outfn)


        
