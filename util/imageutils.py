# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import numpy

def write_pnm_to(img, f, maxval=255):
    if len(img.shape) == 1:
        raise RuntimeError('write_pnm: img is one-dimensional: must be 2 or 3.')
    elif len(img.shape) == 2:
        #pnmtype = 'G'
        pnmcode = 5
        (h,w) = img.shape
    elif len(img.shape) == 3:
        (h,w,planes) = img.shape
        #pnmtype = 'P'
        pnmcode = 6
        if planes != 3:
            raise RuntimeError('write_pnm: img must have 3 planes, not %i' % planes)
    else:
        raise RuntimeError('write_pnm: img must have <= 3 dimensions.')

    if img.max() > maxval:
        print('write_pnm: Some pixel values are > maxval (%i): clipping them.' % maxval)
    if img.min() < 0:
        print('write_pnm: Some pixel values are < 0: clipping them.')
    clipped = img.clip(0, maxval)

    maxval = int(maxval)
    if maxval > 65535:
        raise RuntimeError('write_pnm: maxval must be <= 65535')
    if maxval < 0:
        raise RuntimeError('write_pnm: maxval must be positive')

    f.write('P%i %i %i %i ' % (pnmcode, w, h, maxval))
    if maxval <= 255:
        f.write(img.astype(numpy.uint8).data)
    else:
        f.write(img.astype(numpy.uint16).data)


def write_pnm(img, filename, maxval=255):
    f = open(filename, 'wb')
    write_pnm_to(img, f, maxval)
    f.close()

