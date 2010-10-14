#! /usr/bin/env python

from pylab import *
from numpy import *
from astrometry.util.sip import *
from optparse import OptionParser

def plotDistortion(sip, W, H, ncells, exaggerate=1.):
	'''
	Produces a plot showing the SIP distortion that was found, by drawing
	a grid and distorting it.  Allows exaggeration of the distortion for ease
	of visualization.

	sip -- an astrometry.util.Sip/Tan object
	W, H -- the image size
	ncells -- the approximate number of grid cells to split the image into.
	prefix -- output plot filename prefix.
	exaggerate -- the factor by which to exaggerate the distortion.
	
	'''
	ncells = float(ncells)
	cellsize = sqrt(W * H / ncells)
	nw = int(floor(W / cellsize))
	nh = int(floor(H / cellsize))
	print 'Grid cell size', cellsize
	print 'N cells', nw, 'x', nh
	cx = arange(nw+1) * cellsize + ((W - (nw*cellsize))/2.)
	cy = arange(nh+1) * cellsize + ((H - (nh*cellsize))/2.)

	# pixel step size for grid lines
	step = 50

	xx = arange(-step, W+2*step, step)
	yy = arange(-step, H+2*step, step)

	clf()

	for y in cy:
		dx,dy = [],[]
		for x in xx:
			dxi,dyi = sip.get_distortion(x, y)
			dx.append(dxi)
			dy.append(dyi)
		plot(xx, y*ones_like(xx), 'k-', zorder=10)
		dx = array(dx)
		dy = array(dy)
		if exaggerate != 1:
			dx += (exaggerate * (dx - xx))
			dy += (exaggerate * (dy - y))
		plot(dx, dy, 'r-', zorder=20)

	for x in cx:
		dx,dy = [],[]
		for y in yy:
			dxi,dyi = sip.get_distortion(x, y)
			dx.append(dxi)
			dy.append(dyi)
		plot(x*ones_like(yy), yy, 'k-', zorder=10)
		dx = array(dx)
		dy = array(dy)
		if exaggerate != 1:
			dx += (exaggerate * (dx - x))
			dy += (exaggerate * (dy - yy))
		plot(dx, dy, 'r-', zorder=20)

	
	axis('scaled')
	axis([0, W, 0, H])



if __name__ == '__main__':
	parser = OptionParser('usage: %prog [options] <wcs-filename> <plot-output-filename>')
	parser.add_option('-e', dest='ext', type='int', help='FITS extension to read WCS from (default 0)')
	parser.add_option('-x', dest='exaggerate', type='float', help='Exaggeration factor')
	parser.add_option('-c', dest='cells', type='int', help='Approx. number of pieces to cut image into (default:18)')
	parser.set_defaults(ext=0, cells=18, exaggerate=1.)
	opt,args = parser.parse_args()
	if len(args) != 2:
		parser.print_help()
		sys.exit(-1)

	wcsfn = args[0]
	outfn = args[1]

	wcs = Sip(wcsfn, opt.ext)
	if wcs is None:
		print 'Failed to open WCS file', wcsfn
		sys.exit(-1)

	plotDistortion(wcs, wcs.get_width(), wcs.get_height(), opt.cells, opt.exaggerate)
	savefig(outfn)
