#! /usr/bin/env python
import os
import sys
import re

if __name__ == '__main__':
	try:
		import pyfits
	except ImportError,ie:
		print 'failed to import pyfits:', ie
		me = sys.argv[0]
		#print 'i am', me
		path = os.path.realpath(me)
		#print 'my real path is', path
		utildir = os.path.dirname(path)
		assert(os.path.basename(utildir) == 'util')
		andir = os.path.dirname(utildir)
		assert(os.path.basename(andir) == 'astrometry')
		rootdir = os.path.dirname(andir)
		#print 'adding path', rootdir
		sys.path += [rootdir]

import pyfits

def fits2fits(infile, outfile, verbose):
	"""
	Returns: error string, or None on success.
	"""
	# Read input file.
	fitsin = pyfits.open(infile)
	# Print out info about input file.
	if verbose:
		fitsin.info()
	# Create output list of HDUs
	fitsout = pyfits.HDUList()

	for i, hdu in enumerate(fitsin):
		# verify() fails when a keywords contains invalid characters,
		# so go through the primary header and fix them by converting invalid
		# characters to '_'
		hdr = fitsin[i].header
		cards = hdr.ascardlist()
		# allowed charactors (FITS standard section 5.1.2.1)
		pat = re.compile(r'[^A-Z0-9_\-]')
		for c in cards.keys():
			# new keyword:
			cnew = pat.sub('_', c)
			if (c != cnew):
				if verbose:
					print 'Replacing illegal keyword ', c, ' by ', cnew
				# add the new header card
				hdr.update(cnew, cards[c].value, cards[c].comment, after=c)
				# remove the old one.
				del hdr[c]

		# Fix input header
		fitsin[i].verify('fix')
		# Copy fixed input header to output
		fitsout.append(fitsin[i])

	# Describe output file we're about to write...
	if verbose:
		print 'Outputting:'
		fitsout.info()

	try:
		#fitsout.writeto(outfile, clobber=True)
		fitsout.writeto(outfile, clobber=True, output_verify='warn')
	except pyfits.VerifyError, ve:
		return ('Verification of output file failed: your FITS file is probably too broken to automatically fix.' +
				'  Error message is:' + str(ve))
	return None

def main():
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-v', '--verbose',
					  action='store_true', dest='verbose',
					  help='be chatty')
	(options, args) = parser.parse_args()
	#verbose = options.verbose

	if len(args) != 2:
		print 'Usage: fits2fits.py [--verbose] input.fits output.fits'
		return -1

	infn = args[0]
	outfn = args[1]
	if fits2fits(infn, outfn, options.verbose):
		return -1
	return 0

if __name__ == '__main__':
	sys.exit(main())

