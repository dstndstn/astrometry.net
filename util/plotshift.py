from pylab import *
from numpy import *

from astrometry.libkd import spherematch

def plotshift(ixy, rxy, dcell=50, ncells=18, outfn=None, W=None, H=None):
	# correspondences we could have hit...
	radius = dcell * sqrt(2.)
	print 'ixy', ixy.shape
	print 'rxy', rxy.shape
	assert(rxy.shape[1] == 2)
	assert(ixy.shape[1] == 2)

	ix = ixy[:,0]
	iy = ixy[:,1]

	if W is None:
		W = max(ix)
	if H is None:
		H = max(iy)

	keep = (rxy[:,0] > -dcell) * (rxy[:,0] < W+dcell) * (rxy[:,1] > -dcell) * (rxy[:,1] < H+dcell)
	rxy = rxy[keep]
	print 'Cut to %i ref sources in range' % len(rxy)
	
	cellsize = sqrt(W * H / ncells)
	nw = int(round(W / cellsize))
	nh = int(round(H / cellsize))
	print 'Grid cell size', cellsize
	print 'N cells', nw, 'x', nh
	edgesx = linspace(0, W, nw+1)
	edgesy = linspace(0, H, nh+1)
	print 'Edges:'
	print '  x:', edgesx
	print '  y:', edgesy
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
			print 'cell %i, %i' % (j, i)
			print '%i ref sources' % sum(R)

			if sum(R) > 0:
				(inds,dists) = spherematch.match(rxy[R,:], ixy, radius)
				print 'Found %i matches within %g pixels' % (len(dists), radius)
				ri = inds[:,0]
				# un-cut ref inds...
				ri = (flatnonzero(R))[ri]
				ii = inds[:,1]
				matchx  = rxy[ri,0]
				matchy  = rxy[ri,1]
				matchdx = ix[ii] - matchx
				matchdy = iy[ii] - matchy
				ok = (matchdx >= -dcell) * (matchdx <= dcell) * (matchdy >= -dcell) * (matchdy <= dcell)
				matchdx = matchdx[ok]
				matchdy = matchdy[ok]
				print 'Cut to %i within %g x %g square' % (sum(ok), dcell*2, dcell*2)
			
			# Subplot places plots left-to-right, TOP-to-BOTTOM.
			subplot(nh, nw, 1 + ((nh - i - 1)*nw + j))
			if sum(R) > 0:
				plot(matchdx, matchdy, 'ro', mec='r', mfc='r', ms=5, alpha=0.2)
				plot(matchdx, matchdy, 'ro', mec='r', mfc='none', ms=5, alpha=0.2)

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
		print 'Saving', outfn
		savefig(outfn)
