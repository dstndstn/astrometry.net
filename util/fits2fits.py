#! /usr/bin/env python
import os
import sys
import re

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
				if verbose:
					print 'Replacing illegal keyword ', k, ' by ', knew
				# add the new header card
				hdr.update(knew, cards[k].value, cards[k].comment, after=k)
				# remove the old one.
				del hdr[k]

		# Fix input header
		hdu.verify('fix')

	# Describe output file we're about to write...
	if verbose:
		print 'Outputting:'
		fitsin.info()

	try:
		fitsin.writeto(outfile, clobber=True, output_verify='warn')
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
	#verbose = options.verbose

	if len(args) != 2:
		print 'Usage: fits2fits.py [--verbose] input.fits output.fits'
		return -1

	infn = args[0]
	outfn = args[1]
	errmsg = fits2fits(infn, outfn, verbose=options.verbose, fix_idr=options.fix_idr)
	if errmsg is not None:
		print 'fits2fits.py failed:', errmsg
		return -1
	return 0

if __name__ == '__main__':
	sys.exit(main())

