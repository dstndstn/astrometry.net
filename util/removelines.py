#! /usr/bin/env python
import sys
import logging
import pyfits

from numpy import *
from numpy.random import rand
from pylab import hist, find

## DEBUG
from pylab import *

logging.basicConfig(level=logging.DEBUG)

logverb = logging.debug

def hist_remove_lines(x, binwidth, binoffset, logcut):
    bins = -binoffset + arange(0, max(x)+binwidth, binwidth)
    (counts, thebins, p) = hist(x, bins)

    # We're ignoring empty bins.
    occupied = find(counts > 0)
    noccupied = len(occupied)
    k = (counts[occupied] - 1) 
    mean = sum(k) / float(noccupied)
    logpoisson = k*log(mean) - mean - array([sum(arange(kk)) for kk in k])
    badbins = occupied[logpoisson < logcut]
    badleft = bins[badbins]
    badright = badleft + binwidth

    badpoints = sum(array([(x >= L)*(x < R) for (L,R) in zip(badleft, badright)]), 0)
    return (badpoints == 0)

def removelines(infile, outfile, **kwargs):
    logverb('Reading xylist from file', infile)
    p = pyfits.open(infile)
    xy = p[1].data
    hdr = p[1].header
    x = xy.field('X')
    y = xy.field('Y')

    if False:
        W = int(hdr.get('IMAGEW', '0'))
        if W <= 0:
            W = floor(max(x))
        H = int(hdr.get('IMAGEH', '0'))
        if H <= 0:
            H = floor(max(y))
        logverb('Image width %i, height %i' % (W,H))

    ix = hist_remove_lines(x, 1, 0.5, -100)
    iy = hist_remove_lines(y, 1, 0.5, -100)
    I = ix * iy
    xc = x[I]
    yc = y[I]

    p[1].header.add_history('This xylist was filtered by the "removelines.py" program')
    p[1].header.add_history('to remove horizontal and vertical lines of sources')
    p[1].header.update('REMLINEN', len(x) - len(xc), 'Number of sources removed by "removelines.py"')

    p[1].data = p[1].data[I]
    p.writeto(outfile, clobber=True)

    return 0

if __name__ == '__main__':
	if (len(sys.argv) == 3):
		infile = sys.argv[1]
		outfile = sys.argv[2]
		verbose = False
        rtncode = removelines(infile, outfile)
        sys.exit(rtncode)


