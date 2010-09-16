#! /usr/bin/env python

import sys
from optparse import OptionParser

import matplotlib
matplotlib.use('Agg')

from pylab import *
from numpy import *

from astrometry.libkd import spherematch

def plotshift(ixy, rxy, dcell=50, ncells=18, outfn=None, W=None, H=None, hist=False,
			  nhistbins=21):
			  #histbinsize=None):
	# correspondences we could have hit...
	radius = dcell * sqrt(2.)
	#print 'ixy', ixy.shape
	#print 'rxy', rxy.shape
	assert((len(rxy) == 0) or (rxy.shape[1] == 2))
	assert((len(ixy) == 0) or (ixy.shape[1] == 2))

	ix = ixy[:,0]
	iy = ixy[:,1]

	if W is None:
		W = max(ix)
	if H is None:
		H = max(iy)

	if len(rxy):
		keep = (rxy[:,0] > -dcell) * (rxy[:,0] < W+dcell) * (rxy[:,1] > -dcell) * (rxy[:,1] < H+dcell)
		rxy = rxy[keep]
    #print 'Cut to %i ref sources in range' % len(rxy)
	
	cellsize = sqrt(W * H / ncells)
	nw = int(round(W / cellsize))
	nh = int(round(H / cellsize))
	#print 'Grid cell size', cellsize
	#print 'N cells', nw, 'x', nh
	edgesx = linspace(0, W, nw+1)
	edgesy = linspace(0, H, nh+1)
	#print 'Edges:'
	#print '	 x:', edgesx
	#print '	 y:', edgesy
	if len(rxy) == 0:
		binx = array([])
		biny = array([])
	else:
		binx = digitize(rxy[:,0], edgesx)
		biny = digitize(rxy[:,1], edgesy)
	binx = clip(binx - 1, 0, nw-1)
	biny = clip(biny - 1, 0, nh-1)
	bin = biny * nw + binx

	clf()

	for i in range(nh):
		for j in range(nw):
			thisbin = i * nw + j
			R = (bin == thisbin)
			#print 'cell %i, %i' % (j, i)
			#print '%i ref sources' % sum(R)
			matchdx = []

			if sum(R) > 0:
				(inds,dists) = spherematch.match(rxy[R,:], ixy, radius)
				#print 'Found %i matches within %g pixels' % (len(dists), radius)
				ri = inds[:,0]
				# un-cut ref inds...
				ri = (flatnonzero(R))[ri]
				ii = inds[:,1]
				matchx	= rxy[ri,0]
				matchy	= rxy[ri,1]
				matchdx = ix[ii] - matchx
				matchdy = iy[ii] - matchy
				#print 'All matches:'
				#for dx,dy in zip(matchdx,matchdy):
				#	print '	 %.1f, %.1f' % (dx,dy)
				ok = (matchdx >= -dcell) * (matchdx <= dcell) * (matchdy >= -dcell) * (matchdy <= dcell)
				matchdx = matchdx[ok]
				matchdy = matchdy[ok]
				#print 'Cut to %i within %g x %g square' % (sum(ok), dcell*2, dcell*2)
				#print 'Cut matches:'
				#for dx,dy in zip(matchdx,matchdy):
				#	print '	 %.1f, %.1f' % (dx,dy)
			
			# Subplot places plots left-to-right, TOP-to-BOTTOM.
			subplot(nh, nw, 1 + ((nh - i - 1)*nw + j))

			if len(matchdx) > 0:
				#plot(matchdx, matchdy, 'ro', mec='r', mfc='r', ms=5, alpha=0.2)
				#plot(matchdx, matchdy, 'ro', mec='r', mfc='none', ms=5, alpha=0.2)
				if hist:
					#if histbinsize is None:
					#	histbinsize = dcell / 10.
					edges = linspace(-dcell, dcell, nhistbins+1)
					(H,xe,ye) = histogram2d(matchdx, matchdy, bins=(edges,edges))
					imshow(H.T, extent=(min(xe), max(xe), min(ye), max(ye)),
						   aspect='auto', origin='lower', interpolation='nearest')
					text(dcell, dcell, '%i' % H.max(), color='y',
						 horizontalalignment='right', verticalalignment='top')
						 
				else:
					plot(matchdx, matchdy, 'r.', alpha=0.3)

			if hist:
				axhline(0, color='b', alpha=0.8)
				axvline(0, color='b', alpha=0.8)
			else:
				axhline(0, color='k', alpha=0.5)
				axvline(0, color='k', alpha=0.5)
			if i == 0 and j == 0:
				xticks([-dcell,0,dcell])
				yticks([-dcell,0,dcell])
			else:
				xticks([],[])
				yticks([],[])
			axis('scaled')
			axis([-dcell, dcell, -dcell, dcell])
	if outfn is not None:
		#print 'Saving', outfn
		savefig(outfn)


if __name__ == '__main__':
	from astrometry.util.pyfits_utils import fits_table
	
	parser = OptionParser('usage: %prog [options] <image xy> <reference xy> <plot name>')
	parser.add_option('-X', dest='xcol', help='Name of X column in image table')
	parser.add_option('-Y', dest='ycol', help='Name of Y column in image table')
	parser.add_option('-N', dest='nimage', type='int', help='Cut to the first N image sources')
	parser.add_option('-x', dest='rxcol', help='Name of X column in reference table')
	parser.add_option('-y', dest='rycol', help='Name of Y column in reference table')
	parser.add_option('-n', dest='nref', type='int', help='Cut to the first N reference sources')
	parser.add_option('-c', dest='cells', type='int', help='Approx. number of pieces to cut image into (default:18)')
	parser.add_option('-s', dest='cellsize', type='int', help='Search radius, in pixels (default 50)')
	parser.set_defaults(xcol='X', ycol='Y', nimage=0, cells=0, cellsize=0, rxcol='X', rycol='Y', nref=0)
	opt,args = parser.parse_args()

	if len(args) != 3:
		parser.print_help()
		sys.exit(-1)

	imxy = fits_table(args[0])
	refxy = fits_table(args[1])
	outfn = args[2]

	kwargs = {}
	if opt.cells:
		kwargs['ncells'] = opt.cells
	if opt.cellsize:
		kwargs['dcell'] = opt.cellsize

	ix = imxy.getcolumn(opt.xcol)
	iy = imxy.getcolumn(opt.ycol)
	ixy = vstack((ix,iy)).T
	if opt.nimage:
		ixy = ixy[:opt.nimage,:]

	rx = refxy.getcolumn(opt.rxcol)
	ry = refxy.getcolumn(opt.rycol)
	rxy = vstack((rx,ry)).T
	if opt.nref:
		rxy = rxy[:opt.nref,:]

	plotshift(ixy, rxy, outfn=outfn, **kwargs)
