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

	catfn = 'catalog.fits'
	indexfn = 'index.fits'
	indexid = 999


	usnobhps = healpix_rangesearch(ra, dec, radius, usnob_nside)
	print 'USNO-B healpixes in range:', usnobhps
	allU = []
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
		allU.append(U)

		clf()
		plot(U.ra, U.dec, 'r.')
		savefig('radec-%i.png' % hp)
		clf()
		hist(U.rmag, 20)
		savefig('rmag-%i.png' % hp)

	ra = hstack([U.ra for U in allU])
	dec = hstack([U.dec for U in allU])
	mag = hstack([U.rmag for U in allU])

	C = pyfits.Column
	pyfits.new_table([C(name='ra', format='D', array=ra, unit='deg'),
					  C(name='dec', format='D', array=dec, unit='deg'),
					  C(name='mag', format='E', array=mag, unit='mag'),
					  ]).writeto(catfn, clobber=True)
	print 'Wrote catalog to', catfn
	
	indnside = int(ceil(healpix_nside_for_side_length_arcmin(scale_low)))

	cmd = (('build-index -i %s -o %s -N %i -l %g -u %g -S %s' +
			' -r 1 -j 1 -p 16 -R 8 -L 20 -E -I %i -M') %
		   (catfn, indexfn, indnside, scale_low, scale_high,
			'mag', indexid))
	print 'Running command:'
	print
	print '  ', cmd
	print
	os.system(cmd)

