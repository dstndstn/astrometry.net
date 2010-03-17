import pyfits
from numpy import *

from pylab import *

from astrometry.util.pyfits_utils import *
from astrometry.util.healpix import *
from astrometry.util.starutil_numpy import *


if __name__ == '__main__':

	usnob_pat = 'usnob10_hp%03i.fits'
	usnob_nside = 9

	ra = 352.270
	dec = -2.921
	# deg
	radius = 1

	# images are 6.8' x 13.9'
	scale_low = 4
	scale_high = 5.6
	
	minmag = 18


	usnobhps = healpix_rangesearch(ra, dec, radius, usnob_nside)
	print 'USNO-B healpixes in range:', usnobhps
	for hp in usnobhps:
		usnobfn = usnob_pat % hp
		print 'USNOB filename:', usnobfn
		U = table_fields(usnobfn)
		I = (U.num_detections >= 2)
		I = logical_and(I, degrees_between(ra, dec, U.ra, U.dec) < radius)
		print '%i USNOB stars within range.' % sum(I)
		U = U[I]
		# 1 is first-epoch red
		# 3 is second-epoch red
		epoch1 = U.field_1 > 0
		epoch2 = U.field_3 > 0
		nmag = where(epoch1, 1, 0) + where(epoch2, 1, 0)
		avgmag = where(epoch1, U.magnitude_1, 0) + where(epoch2, U.magnitude_3, 0)
		avgmag /= nmag
		# might demand existence in both epochs...
		I = logical_and(nmag > 0, avgmag >= minmag)
		print '%i USNOB stars with R mag > %g.' % (sum(I), minmag)
		U = U[I]
		U.rmag = avgmag[I]

		clf()
		plot(U.ra, U.dec, 'r.')
		savefig('radec-%i.png' % hp)
		clf()
		hist(U.rmag, 20)
		savefig('rmag-%i.png' % hp)
		
	#usnobfn =
