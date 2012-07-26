#! /usr/bin/env python
import os
import sys
import logging
from optparse import OptionParser

if __name__ == '__main__':
	try:
		import pyfits
		import astrometry
		from astrometry.util.shell import shell_escape
		from astrometry.util.filetype import filetype_short
	except ImportError:
		me = sys.argv[0]
		path = os.path.realpath(me)
		utildir = os.path.dirname(path)
		assert(os.path.basename(utildir) == 'util')
		andir = os.path.dirname(utildir)
		rootdir = os.path.dirname(andir)
		sys.path.insert(1, andir)
		sys.path.insert(2, rootdir)

import numpy
import pyfits
from numpy import *
from numpy.random import rand
from astrometry.util.pyfits_utils import *

def uniformize(infile, outfile, n, xcol='X', ycol='Y', **kwargs):
	p = pyfits.open(infile)
	xy = p[1].data
	if xy is None:
		print 'No sources'
		pyfits_writeto(p, outfile)
		return
	hdr = p[1].header
	x = xy.field(xcol)
	y = xy.field(ycol)

	# use IMAGEW,H, or compute bounds?
	#  #$)(*&%^ NaNs in LSST source positions.  Seriously, WTF!
	I = logical_and(isfinite(x), isfinite(y))
	if not all(I):
		print '%i source positions are not finite.' % sum(logical_not(I))
		x = x[I]
		y = y[I]
	
	W = max(x) - min(x)
	H = max(y) - min(y)
	if W == 0 or H == 0:
		print 'Area of the rectangle enclosing all image sources: %i x %i' % (W,H)
		pyfits_writeto(p, outfile)
		return
	NX = int(max(1, round(W / sqrt(W*H / float(n)))))
	NY = int(max(1, round(n / float(NX))))
	print 'Uniformizing into %i x %i bins' % (NX, NY)
	print 'Image bounds: x [%g,%g], y [%g,%g]' % (min(x),max(x),min(y),max(y))

	ix = (clip(floor((x - min(x)) / float(W) * NX), 0, NX-1)).astype(int)
	iy = (clip(floor((y - min(y)) / float(H) * NY), 0, NY-1)).astype(int)
	#print ix, iy
	assert(all(ix >= 0))
	assert(all(ix < NX))
	assert(all(iy >= 0))
	assert(all(iy < NY))
	I = iy * NX + ix
	assert(all(I >= 0))
	assert(all(I < NX*NY))
	#print 'len(I):', len(I)
	#print I.shape
	bins = [[] for i in range(NX*NY)]
	for j,i in enumerate(I):
		bins[int(i)].append(j)
	#print bins
	maxlen = max([len(b) for b in bins])
	J = []
	for i in range(maxlen):
		thisrow = []
		for b in bins:
			if i >= len(b):
				continue
			thisrow.append(b[i])
			# J.append(b[i])
		thisrow.sort()
		J += thisrow

	J = array(J)
	#print 'len(J):', len(J)
	p[1].header.add_history('This xylist was filtered by the "uniformize.py" program')
	p[1].data = p[1].data[J]
	pyfits_writeto(p, outfile)
	return 0


if __name__ == '__main__':
	parser = OptionParser(usage='%prog [options] <input-xylist> <output-xylist>')

	parser.add_option('-X', dest='xcol', help='Name of X column in input table')
	parser.add_option('-Y', dest='ycol', help='Name of Y column in input table')
	parser.add_option('-n', dest='n', type='int', help='Number of boxes, approximately')
	parser.set_defaults(xcol='X', ycol='Y', n=10)
	(opt, args) = parser.parse_args()
	
	if len(args) != 2:
		parser.print_help()
		print
		print 'Got arguments:', args
		sys.exit(-1)

	infile = args[0]
	outfile = args[1]
	rtncode = uniformize(infile, outfile, opt.n, xcol=opt.xcol, ycol=opt.ycol)
	sys.exit(rtncode)

