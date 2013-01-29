from numpy import sin, atleast_1d, zeros, logical_and
from math import pi
import numpy as np

def point_in_poly(x, y, poly):
	'''
	Performs a point-in-polygon test for numpy arrays of *x* and *y*
	values, and a polygon described as 2-d numpy array.

	poly: N x 2 array

	Returns a numpy array of bools.
	'''
	inside = np.zeros(np.atleast_1d(x).shape, bool)
	for i in range(len(poly)):
		j = (i-1 + len(poly)) % len(poly)
		xi,xj = poly[i,0], poly[j,0]
		yi,yj = poly[i,1], poly[j,1]
		I = np.logical_and(
			np.logical_or(np.logical_and(yi <= y, y < yj),
						  np.logical_and(yj <= y, y < yi)),
			x < (xi + ((xj - xi) * (y - yi) / (yj - yi))))
		inside[I] = np.logical_not(inside[I])
	return inside

def lanczos_filter(order, x):
	x = np.atleast_1d(x)
	nz = np.logical_and(x != 0., np.logical_and(x < order, x > -order))
	nz = np.flatnonzero(nz)
									   
	#filt = np.zeros(len(x), float)
	filt = np.zeros(x.shape, dtype=float)
	#filt[nz] = order * sin(pi * x[nz]) * sin(pi * x[nz] / order) / ((pi * x[nz])**2)
	pinz = pi * x.flat[nz]
	filt.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
	filt[x == 0] = 1.
	#filt[x >  order] = 0.
	#filt[x < -order] = 0.
	return filt

# Given a range of integer coordinates that you want to, eg, cut out
# of an image, [xlo, xhi], and bounds for the image [xmin, xmax],
# returns the range of coordinates that are in-bounds, and the
# corresponding region within the desired cutout.
def get_overlapping_region(xlo, xhi, xmin, xmax):
	if xlo > xmax or xhi < xmin or xlo > xhi or xmin > xmax:
		return ([], [])

	assert(xlo <= xhi)
	assert(xmin <= xmax)
	
	xloclamp = max(xlo, xmin)
	Xlo = xloclamp - xlo

	xhiclamp = min(xhi, xmax)
	Xhi = Xlo + (xhiclamp - xloclamp)

	#print 'xlo, xloclamp, xhiclamp, xhi', xlo, xloclamp, xhiclamp, xhi
	assert(xloclamp >= xlo)
	assert(xloclamp >= xmin)
	assert(xloclamp <= xmax)
	assert(xhiclamp <= xhi)
	assert(xhiclamp >= xmin)
	assert(xhiclamp <= xmax)
	#print 'Xlo, Xhi, (xmax-xmin)', Xlo, Xhi, xmax-xmin
	assert(Xlo >= 0)
	assert(Xhi >= 0)
	assert(Xlo <= (xhi-xlo))
	assert(Xhi <= (xhi-xlo))

	return (slice(xloclamp, xhiclamp+1), slice(Xlo, Xhi+1))
