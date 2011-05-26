#! /usr/bin/env python

import os
import sys
import pyfits
from astrometry.util.util import *


def main():
	from optparse import OptionParser
	parser = OptionParser(usage='%prog [options] <ra> <dec> <fpC> <cutout>')
	parser.add_option('-s', '--size', dest='size', type=int, default=300,
					  help='Size of cutout in pixels')
	(opt, args) = parser.parse_args()
	if len(args) != 4:
		parser.print_help()
		print 'Got arguments:', args
		sys.exit(-1)
	# parse RA,Dec.
	ra = float(args[0])
	dec = float(args[1])
	fpC = args[2]
	cutout = args[3]

	if not os.path.exists(fpC):
		print 'Input fpC', fpC, 'does not exist'
		sys.exit(-1)

	wcs = Tan(fpC)
	x,y = wcs.radec2pixelxy(ra, dec)
	x,y = int(x),int(y)
	print 'x,y', x,y
	dl = opt.size / 2
	dh = opt.size - dl
	os.system('imcopy %s"[%i:%i,%i:%i]" !%s' % (fpC, max(0, x-dl), x+dh, max(0, y-dl), y+dh, cutout))
	



if __name__ == '__main__':
	main()
	
