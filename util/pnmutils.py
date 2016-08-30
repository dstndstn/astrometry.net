# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from numpy import *

# Writes a numpy array as PGM to the given file handle.  The numpy
# array will be converted to integer values.  If maxval=255, 8-bit
# ints; if maxval>255, 16-bit ints.
def write_pgm(x, f, maxval=255):
    (h,w) = x.shape
    if maxval >= 65536:
        raise RuntimeError('write_pgm: maxval must be < 65536')

    f.write('P5 %i %i %i\n' % (w, h, maxval))
    if maxval <= 255:
        f.write(getbuffer(x.astype(uint8)))
    else:
        f.write(getbuffer(x.astype(uint16)))


# Writes a numpy array as PPM to the given file handle.  The numpy
# array will be converted to integer values.  If maxval=255, 8-bit
# ints; if maxval>255, 16-bit ints.
def write_ppm(x, f, maxval=255):
    (h,w,d) = x.shape
    if maxval >= 65536:
        raise RuntimeError('write_pgm: maxval must be < 65536')
    if d != 3:
        raise Exception('array must have shape H x W x 3; got ' + x.shape)

    f.write('P6 %i %i %i\n' % (w, h, maxval))
    if maxval <= 255:
        f.write(getbuffer(x.astype(uint8)))
    else:
        f.write(getbuffer(x.astype(uint16)))
