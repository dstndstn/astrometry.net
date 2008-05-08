import sys

from pylab import *
from numpy import *

import pyfits

# Given an image and an xylist (including estimated image sigma),
# look at a cutout around each source position, add noise, and recompute
# dcen3x3 to find the noisy peak.


if __name__ == '__main__':
    imgfn = sys.argv[1]
    xyfn = sys.argv[2]

    print 'FITS Image', imgfn
    print 'xylist', xyfn

    p = pyfits.open(imgfn)
    I = p[1].data
    print 'Image is', I.shape

    p.pyfits.open(xyfn)
    xy = p[1].data
    x = xy.field('X')
    y = xy.field('Y')
    print 'Sources: %i', len(x)
