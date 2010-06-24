#! /usr/bin/env python
import sys
from optparse import OptionParser

import pyfits
from numpy import *

from astrometry.util.pyfits_utils import *
from astrometry.util.healpix import *
from astrometry.util.starutil_numpy import *

def get_usnob_sources(ra, dec, radius=1, basefn=None):
	usnob_nside = 9

	if basefn is None:
		usnob_pat = 'usnob10_hp%03i.fits'
	else:
		usnob_pat = basefn

	usnobhps = healpix_rangesearch(ra, dec, radius, usnob_nside)
	print 'USNO-B healpixes in range:', usnobhps
	allU = None
	for hp in usnobhps:
		usnobfn = usnob_pat % hp
		print 'USNOB filename:', usnobfn
		U = table_fields(usnobfn)
		I = (degrees_between(ra, dec, U.ra, U.dec) < radius)
		print '%i USNOB stars within range.' % sum(I)
		U = U[I]
		if allU is None:
			allU = U
		else:
			allU.append(U)
	return allU

if __name__ == '__main__':
	parser = OptionParser(usage='%prog [options] <ra> <dec> <output-filename>')

	parser.add_option('-r', dest='radius', type='float', help='Search radius, in deg (default 1 deg)')
	parser.add_option('-b', dest='basefn', help='Base filename of USNO-B FITS files (default: usnob10_hp%03i.fits)')
	parser.set_defaults(radius=1.0, basefn=None)

	(opt, args) = parser.parse_args()
	if len(args) != 3:
		parser.print_help()
		print
		print 'Got extra arguments:', args
		sys.exit(-1)

	# parse RA,Dec.
	ra = float(args[0])
	dec = float(args[1])
	outfn = args[2]

	# ugh!
	opts = {}
	for k in ['radius', 'basefn']:
		opts[k] = getattr(opt, k)

	X = get_usnob_sources(ra, dec, **opts)
	print 'Got %i USNO-B sources.' % len(X)

	print 'Applying cuts...'
	# USNO-B sources (not Tycho-2)
	I = (X.num_detections >= 2)
	# no diffraction spikes
	I = logical_and(I, logical_not(X.flags[:,0]))
	X = X[I]
	print '%i pass cuts' % len(X)

	#from pylab import *
	#clf()
	#plot(X.ra, X.dec, 'r.')
	#savefig('radec.png')

	# Compute average R and B mags.
	epoch1 = (X.field_1 > 0)
	epoch2 = (X.field_3 > 0)
	nmag = where(epoch1, 1, 0) + where(epoch2, 1, 0)
	summag = where(epoch1, X.magnitude_1, 0) + where(epoch2, X.magnitude_3, 0)
	X.r_mag = where(nmag == 0, 0, summag / nmag)
	# B
	epoch1 = (X.field_0 > 0)
	epoch2 = (X.field_2 > 0)
	nmag = where(epoch1, 1, 0) + where(epoch2, 1, 0)
	summag = where(epoch1, X.magnitude_0, 0) + where(epoch2, X.magnitude_2, 0)
	X.b_mag = where(nmag == 0, 0, summag / nmag)

	print 'Writing to', outfn
	X.write_to(outfn)
	
