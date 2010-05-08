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
#from astrometry.util.pyfits_utils import *

def uniformize(infile, outfile, n, xcol='X', ycol='Y', **kwargs):
	p = pyfits.open(infile)
	xy = p[1].data
	hdr = p[1].header
	x = xy.field(xcol)
	y = xy.field(ycol)

	# use IMAGEW,H, or compute bounds?
	W = max(x) - min(x)
	H = max(y) - min(y)
	NX = int(max(1, round(W / sqrt(W*H / float(n)))))
	NY = int(max(1, round(n / float(NX))))
	print 'Uniformizing into %i x %i bins' % (NX, NY)

	ix = (clip(floor((x - min(x)) / float(W) * NX), 0, NX-1)).astype(int)
	iy = (clip(floor((y - min(y)) / float(H) * NY), 0, NY-1)).astype(int)
	I = iy * NX + ix
	#print 'len(I):', len(I)
	#print I.shape
	bins = [[] for i in range(NX*NY)]
	for j,i in enumerate(I):
		bins[int(i)].append(j)
	#print bins
	maxlen = max([len(b) for b in bins])
	J = []
	for i in range(maxlen):
		for b in bins:
			if i >= len(b):
				continue
			J.append(b[i])
	J = array(J)
	#print 'len(J):', len(J)
	p[1].header.add_history('This xylist was filtered by the "uniformize.py" program')
	p[1].data = p[1].data[J]
	p.writeto(outfile, clobber=True)
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

