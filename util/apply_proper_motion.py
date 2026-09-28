#! /usr/bin/env python3

""" Utility to apply proper motion to an astronomical catalog in FITS format.
    Will create a new table containing all data of the original table
    as well as two additional columns, the modified RA and DEC values.

    This utility assumes that proper motion coordinates
    are given in ra*cos(dec) and dec.
"""

import argparse
import sys
import os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', '--infile', required=True)
parser.add_argument('-o', '--outfile', required=True)
parser.add_argument('-a', '--ra', required=True, help='RA column name')
parser.add_argument('-d', '--dec', required=True, help='DEC column name')
parser.add_argument('--pmra', required=True,
                    help='RA proper motion column name')
parser.add_argument('--pmde', required=True,
                    help='DEC proper motion column name')
parser.add_argument('--t0', required=True,
                    help='Original date (YYYY-MM-DD)')
parser.add_argument('--t1', required=True,
                    help='Destination date (YYYY-MM-DD)')
parser.add_argument('--clean', action='store_true',
                    help='If provided, will remove all rows '
                         'from the generated table '
                         'where any value is missing')
parser.add_argument('--overwrite', action='store_true',
                    help='If provided, will overwrite the output file '
                         'if it already exists')

args = parser.parse_args()

if not args.overwrite and os.path.exists(args.outfile):
    sys.stderr.write('%s already exists. To overwrite, pass --overwrite.\n'
                     % args.outfile)
    exit(-1)

t = Table.read(args.infile)

if args.clean:
    if t.masked:
        t = t[~tbl.mask.any(axis=1)]
    else:
        keep = np.ones(len(t), dtype=bool)
        for col in t.itercols():
            if np.issubdtype(col.dtype, np.number):
                keep &= ~np.isnan(col)
            elif col.dtype.kind in ("U", "S"):
                keep &= col != ""
                keep &= col != " "
        t = t[keep]

sys.stderr.write(f'RA and DEC units are: {t[args.ra].unit}, {t[args.dec].unit}\n')
sys.stderr.write(f'pmRA and pmDE units are: {t[args.pmra].unit}, {t[args.pmde].unit}\n')
sys.stderr.write('If you encounter errors below, '
                 'it might be because these units aren\'t deg and mas/yr, respectively\n')

coords = SkyCoord(
    ra       = t[args.ra].quantity,
    dec      = t[args.dec].quantity,
    pm_ra_cosdec = t[args.pmra].quantity,
    pm_dec       = t[args.pmde].quantity,
    obstime = Time(args.t0, scale='utc'),
    frame   = 'icrs'
)

target_epoch = Time(args.t1, scale='utc')
new_coords = coords.apply_space_motion(new_obstime=target_epoch)

t['RA_' + args.t1] = new_coords.ra.deg
t['DEC_' + args.t1] = new_coords.dec.deg

t.write(args.outfile, overwrite=args.overwrite)
