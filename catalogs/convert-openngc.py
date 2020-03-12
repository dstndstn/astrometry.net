# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

# Generates FITS tables from CSV lists of OpenNGC entries and names.

from __future__ import print_function

import csv

from astrometry.util.fits import fits_table
import numpy as np


def convert_openngc_entries():
    entries = []

    with open('openngc-entries.csv') as f:
        for is_ngc, num, ra, dec, size in csv.reader(f, delimiter=';'):
            is_ngc = (is_ngc == '1')
            num = int(num)
            ra = float(ra) if ra else 0.0
            dec = float(dec) if dec else 0.0

            # Convert from diameter in arcmins to radius in degrees.
            radius = float(size) / (2.0 * 60.0) if size else 0.0

            entries.append({
                'is_ngc': is_ngc,
                'ra': ra,
                'dec': dec,
                'radius': radius,
                'num': num,
            })

    T = fits_table()
    for key in ['is_ngc', 'ra', 'dec', 'radius', 'num']:
        T.set(key, [x[key] for x in entries])

    T.to_np_arrays()

    T.name = np.array(['NGC %i' % n if isngc else 'IC %i' % n
                       for n, isngc in zip(T.num, T.is_ngc)])

    for key in ['ra', 'dec', 'radius']:
        T.set(key, T.get(key).astype(np.float32))
    T.num = T.num.astype(np.int16)

    units_dict = {
        'ra': 'deg',
        'dec': 'deg',
        'radius': 'deg',
    }

    NGC = T[T.is_ngc]
    NGC.rename('num', 'ngcnum')
    NGC.delete_column('is_ngc')
    units = [units_dict.get(c, '') for c in NGC.get_columns()]
    NGC.writeto('openngc-ngc.fits', units=units)

    IC = T[np.logical_not(T.is_ngc)]
    IC.rename('num', 'icnum')
    IC.delete_column('is_ngc')
    units = [units_dict.get(c, '') for c in IC.get_columns()]
    IC.writeto('openngc-ic.fits', units=units)


def convert_openngc_names():
    names = []

    with open('openngc-names.csv') as f:
        for is_ngc, num, name in csv.reader(f, delimiter=';'):

            is_ngc = bool(is_ngc)

            num = int(num)

            identifier = '%s%d' % ('' if is_ngc else 'I', num)

            names.append({
                'Object': name,
                'Name': identifier,
            })

    T = fits_table()
    for key in ['Object', 'Name']:
        T.set(key, [x[key] for x in names])
    T.writeto('openngc-names.fits')


if __name__ == '__main__':
    convert_openngc_entries()
    convert_openngc_names()
