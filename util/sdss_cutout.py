#! /usr/bin/env python

import os
import sys
import pyfits
from astrometry.util.util import *
import pyfits
from astrometry.util.sdss_noise import *
from astrometry.util.sdss_psfield import *


def main():
	from optparse import OptionParser
	parser = OptionParser(usage='%prog [options] <ra> <dec> <fpC> <cutout> [<fpM> <psField> <invvar-cutout>]')
	parser.add_option('-s', '--size', dest='size', type=int, default=300,
					  help='Size of cutout in pixels')
	parser.add_option('-b', '--band', dest='band', default='r',
					  help='Band (u,g,r,i,z)')
	(opt, args) = parser.parse_args()
	if len(args) not in [4,7]:
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

	if len(args) == 7:
		cutout(fpC, ra, dec, opt.size, cutout)
	else:
		cutout(fpC, ra, dec, opt.size, cutout, args[4], args[5], args[6], opt.band)


def cutout(fpC, ra, dec, size, cutout, fpMfn=None, psFieldfn=None, invvarfn=None, band=None):

	wcs = Tan(fpC, 0)
	x,y = wcs.radec2pixelxy(ra, dec)
	x,y = int(x),int(y)
	print 'x,y', x,y
	dl = size / 2
	dh = size - dl
	# ??
	xlo,xhi = max(0, x-dl), min(2048-1, x+dh-1)
	ylo,yhi = max(0, y-dl), min(1489-1, y+dh-1)
	os.system('imcopy %s"[%i:%i,%i:%i]" !%s' %
			  (fpC, xlo, xhi, ylo, yhi, cutout))

	if invvarfn is None:
		   return
		
	bandnum = 'ugriz'.index(band)

	fpc = pyfits.open(fpC)[0].data.astype(float)
	fpM = pyfits.open(fpMfn)
	(gain, darkvar, sky, skyerr) = sdss_psfield_noise(psFieldfn, band=bandnum)

	invvar = sdss_noise_invvar(fpc, fpM, xlo, xhi, ylo, yhi,
							   gain, darkvar, sky, skyerr)
	print invvar.shape
	#print 'x', xlo, xhi
	#print 'y', ylo, yhi
	#invvar = invvar[ylo:yhi, xlo:xhi]
	#print invvar.shape
	pyfits.writeto(invvarfn, invvar, clobber=True)


if __name__ == '__main__':
	main()
	
