#! /usr/bin/env python
from astrometry.util.sip import *

from optparse import OptionParser

if __name__ == '__main__':
	parser = OptionParser('usage: %prog [options] <outfn>')
	#parser.add_option('-o', dest='ra', type='float', help='RA (deg)')
	parser.add_option('-r', '--ra', dest='ra', type='float', help='RA (deg)')
	parser.add_option('-d', '--dec', dest='dec', type='float', help='Dec (deg)')
	parser.add_option('-s', '--size', dest='size', type='float', help='Field width (deg)')
	parser.add_option('-W', '--width', dest='w', type='int', help='Image width')
	parser.add_option('-H', '--height', dest='h', type='int', help='Image height')
	# crpix at image middle
	parser.set_defaults(ra=None, dec=None, size=None, w=None, h=None)
	opt,args = parser.parse_args()

	wcs = Tan()
	wcs.crval[0] = opt.ra
	wcs.crval[1] = opt.dec
	wcs.imagew = opt.w
	wcs.imageh = opt.h
	wcs.crpix[0] = 0.5 + (opt.w / 2.)
	wcs.crpix[1] = 0.5 + (opt.h / 2.)
	pixscale = opt.size / opt.w
	wcs.cd[0] = -pixscale
	wcs.cd[1] = 0
	wcs.cd[2] = 0
	wcs.cd[3] = -pixscale

	wcs.write_to_file(args[0])

