#! /usr/bin/env python
import os
import sys
import re
import logging

if __name__ == '__main__':
	# According to the python sys.path documentation, the directory containing
	# the main script appears as sys.path[0].
	utildir = sys.path[0]
	assert(os.path.basename(utildir) == 'util')
	andir = os.path.dirname(utildir)
	#assert(os.path.basename(andir) == 'astrometry')
	rootdir = os.path.dirname(andir)
	# Here we put the "astrometry" and "astrometry/.." directories at the front
	# of the path: astrometry to pick up pyfits, and .. to pick up astrometry itself.
	sys.path.insert(1, andir)
	sys.path.insert(2, rootdir)
	import pyfits

import pyfits
from astrometry.util.pyfits_utils import pyfits_writeto

def fits2fits(infile, outfile, verbose=False, fix_idr=False):
	"""
	Returns: error string, or None on success.
	"""
	if fix_idr:
		from astrometry.util.fix_sdss_idr import fix_sdss_idr

	# Read input file.
	fitsin = pyfits.open(infile)
	# Print out info about input file.
	if verbose:
		fitsin.info()

	for i, hdu in enumerate(fitsin):
		if fix_idr:
			hdu = fitsin[i] = fix_sdss_idr(hdu)
		# verify() fails when a keywords contains invalid characters,
		# so go through the primary header and fix them by converting invalid
		# characters to '_'
		hdr = hdu.header
		cards = hdr.ascardlist()
		# allowed characters (FITS standard section 5.1.2.1)
		pat = re.compile(r'[^A-Z0-9_\-]')
		for k in cards.keys():
			# new keyword:
			knew = pat.sub('_', k)
			if k != knew:
				logging.debug('Replacing illegal keyword %s by %s' % (k, knew))
				# add the new header card
				# it seems pyfits is not clever enough to notice this...
				if len(knew) > 8:
					knew = 'HIERARCH ' + knew
				hdr.update(knew, cards[k].value, cards[k].comment, after=k)
				# remove the old one.
				del hdr[k]

		# Fix input header
		hdu.verify('fix')

		# UGH!  Work around stupid pyfits handling of scaled data...
		# (it fails to round-trip scaled data correctly!)
		bzero = hdr.get('BZERO', None)
		bscale = hdr.get('BSCALE', None)
		if bzero is not None and bscale is not None:
			logging.debug('Scaling to bzero=%g, bscale=%g' % (bzero, bscale))
			hdu.scale('int16', '', bscale, bzero)

	# Describe output file we're about to write...
	if verbose:
		print 'Outputting:'
		fitsin.info()

	try:
		pyfits_writeto(fitsin, outfile, output_verify='warn')
	except pyfits.VerifyError, ve:
		return ('Verification of output file failed: your FITS file is probably too broken to automatically fix.' +
				'  Error message is:' + str(ve))
	fitsin.close()
	return None

def main():
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-v', '--verbose',
					  action='store_true', dest='verbose',
					  help='be chatty')
	parser.add_option('-s', '--fix-sdss',
					  action='store_true', dest='fix_idr',
					  help='fix SDSS idR files')
	(options, args) = parser.parse_args()

	if len(args) != 2:
		print 'Usage: fits2fits.py [--verbose] input.fits output.fits'
		return -1

	logformat = '%(message)s'
	if options.verbose:
		logging.basicConfig(level=logging.DEBUG, format=logformat)
	else:
		logging.basicConfig(level=logging.INFO, format=logformat)
	logging.raiseExceptions = False

	infn = args[0]
	outfn = args[1]
	errmsg = fits2fits(infn, outfn, fix_idr=options.fix_idr,
					   verbose=options.verbose)
	if errmsg is not None:
		print 'fits2fits.py failed:', errmsg
		return -1
	return 0

if __name__ == '__main__':
	sys.exit(main())

