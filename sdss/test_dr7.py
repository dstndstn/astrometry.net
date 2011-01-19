import os

import numpy as np

import astrometry.sdss
from astrometry.sdss import DR7

if __name__ == '__main__':
	sdss = DR7()

	testdata = os.path.join(os.path.dirname(astrometry.sdss.__file__),
							'testdata')
	print 'Using test data dir:', testdata

	sdss.setBasedir(testdata)

	tsfield = sdss.readTsField(2830, 6, 398, 41)

	rband,iband = 2,3

	asr = tsfield.getAsTrans(rband)
	asi = tsfield.getAsTrans(iband)

	x,y = 0,0
	color = 0.

	rdr = asr.pixel_to_radec(x, y, color)
	rdi = asi.pixel_to_radec(x, y, color)
	print 'rdr', rdr
	print 'rdi', rdi
