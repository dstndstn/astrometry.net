# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
from optparse import OptionParser
from astrometry.util.fits import *

def match_to_wcs(matchfn, row, wcsfn, imgw=0, imgh=0):
    m = table_fields(matchfn)
    
    wcs = pyfits.PrimaryHDU()
    h = wcs.header

    hdrs = [
        ('CTYPE1', 'RA---TAN', None),
        ('CTYPE2', 'DEC--TAN', None),
        ('WCSAXES', 2, None),
        ('EQUINOX', 2000.0, 'Equatorial coordinates definition (yr)'),
        ('LONPOLE', 180.0, None),
        ('LATPOLE', 0.0, None),
        ('CUNIT1', 'deg', 'X pixel scale units'),
        ('CUNIT2', 'deg', 'Y pixel scale units'),
        ('CRVAL1', m.crval[row][0], 'RA  of reference point'),
        ('CRVAL2', m.crval[row][1], 'DEC of reference point'),
        ('CRPIX1', m.crpix[row][0], 'X reference pixel'),
        ('CRPIX2', m.crpix[row][1], 'Y reference pixel'),
        ('CD1_1', m.cd[row][0], 'Transformation matrix'),
        ('CD1_2', m.cd[row][1], None),
        ('CD2_1', m.cd[row][2], None),
        ('CD2_2', m.cd[row][3], None),
        ]
    if imgw:
        hdrs.append(('IMAGEW', imgw, 'Image width,  in pixels.'))
    if imgh:
        hdrs.append(('IMAGEH', imgh, 'Image height, in pixels.'))

    for (k,v,c) in hdrs:
        h.update(k, v, c)
    pyfits_writeto(wcs, wcsfn)


if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <match-input-file> <wcs-output-file>')

    parser.add_option('-r', dest='row', help='Row of the match file to take (default 0)', type='int')
    parser.add_option('-W', dest='imgw', help='Image width', type='float')
    parser.add_option('-H', dest='imgh', help='Image height', type='float')

    parser.set_defaults(row=0, imgw=0, imgh=0)

    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        print()
        print('Need args <match-input-file> and <wcs-output-file>')
        sys.exit(-1)

    matchfn = args[0]
    wcsfn = args[1]

    match_to_wcs(matchfn, options.row, wcsfn, options.imgw, options.imgh)

