# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import sys
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
import numpy

from astrometry.util.fits import pyfits_writeto

def is_sdss_idr(hdu):
    hdr = hdu.header
    return ((hdr.get('SIMPLE', True) == False)
            and ('SDSS' in hdr)
            and ('UNSIGNED' in hdr))
            #and hdr.get('SDSS', None) == pyfits.UNDEFINED
            #and hdr.get('UNSIGNED', None) == pyfits.UNDEFINED)

def is_sdss_idr_file(infile):
    p = pyfits.open(infile)
    rtn = is_sdss_idr(p[0])
    p.close()
    return rtn

# Takes a pyfits HDU object (which is unchanged) and returns a new
# pyfits object containing the fixed file.
def fix_sdss_idr(hdu):
    hdr = hdu.header.copy()

    if hdr.get('SIMPLE', True):
        print('SIMPLE = T: not an SDSS idR file.')
        return hdu
    print('Setting SIMPLE = True')
    hdr.remove('SIMPLE')
    hdr.set('SIMPLE', True, 'FITS compliant (via fix-sdss-idr.py)')

    if 'SDSS' in hdr:
        print('Setting SDSS = True')
        hdr.remove('SDSS')
        hdr.set('SDSS', True, 'SDSS (via fix-sdss-idr.py)')
    else:
        print('No SDSS header card: not an SDSS idR file.')
        return hdu

    if 'UNSIGNED' in hdr:
        print('Setting UNSIGNED = True')
        hdr.remove('UNSIGNED')
        hdr.set('UNSIGNED', True, 'SDSS unsigned ints')
    else:
        print('No UNSIGNED header card: not an SDSS idR file.')
        return hdu

    #hdr._updateHDUtype()
    #newhdu = hdr._hdutype(data=pyfits.DELAYED, header=hdr)
    newhdu = pyfits.PrimaryHDU(data=pyfits.DELAYED, header=hdr)
    ## HACK - encapsulation violation
    newhdu._file = hdu._file
    #newhdu._ffile = hdu._ffile
    newhdu._datLoc = hdu._datLoc

    newhdu.data = newhdu.data.astype(numpy.int32)
    newhdu.data[newhdu.data < 0] += 2**16
    print('data type:', newhdu.data.dtype)
    print('data range:', newhdu.data.min(), 'to', newhdu.data.max())
    return newhdu

def fix_sdss_idr_file(infile, outfile):
    print('Reading', infile)
    newhdu = fix_sdss_idr(pyfits.open(infile)[0])
    print('Writing', outfile)
    pyfits_wireto(newhdu, outfile)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <input-fits-file> <output-fits-file>' % sys.argv[0])
        sys.exit(-1)
    infile = sys.argv[1]
    outfile = sys.argv[2]

    fix_sdss_idr_file(infile, outfile)
    sys.exit(0)
