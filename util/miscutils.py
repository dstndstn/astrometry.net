from math import pi
import numpy as np

def polygons_intersect(poly1, poly2):
	'''
	Determines whether the given 2-D polygons intersect.

	poly1, poly2: np arrays with shape (N,2)
	'''

	# Check whether any points in poly1 are inside poly2,
	# or vice versa.
	for (px,py) in poly1:
		if point_in_poly(px,py, poly2):
			return True
	for (px,py) in poly2:
		if point_in_poly(px,py, poly1):
			return True

	# Check for intersections between line segments.  O(n^2) brutish
	N1 = len(poly1)
	N2 = len(poly2)

	for i in range(N1):
		for j in range(N2):
			if line_segments_intersect(poly1[i % N1, :], poly1[(i+1) % N1, :],
									   poly2[j % N2, :], poly2[(j+1) % N2, :]):
				return True
	return False
	

def line_segments_intersect((x1,y1), (x2,y2), (x3,y3), (x4,y4)):
	'''
	Determines whether the two given line segments intersect;

	(x1,y1) to (x2,y2)
	and 
	(x3,y3) to (x4,y4)
	'''
	x,y = intersection((x1,y1),(x2,y2),(x3,y3),(x4,y4))
	if x1 == x2:
		p1,p2 = y1,y2
		p = y
	else:
		p1,p2 = x1,x2
		p = x

	return (p >= min(p1,p2)) and (p <= max(p1,p2))


def intersection((x1,y1), (x2,y2), (x3,y3), (x4,y4)):
	'''
	Determines the point where the lines described by
	(x1,y1) to (x2,y2)
	and 
	(x3,y3) to (x4,y4)
	intersect.

	Note that this may be beyond the endpoints of the line segments.

	Probably raises an exception if the lines are parallel, or does
	something numerically crazy.
	'''
	# copy-n-paste from Wikipedia, latex->python -- woo!
	px = (((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) /
		  ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))

	py = (((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) /
		  ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)))
	return px,py


def point_in_poly(x, y, poly):
	'''
	Performs a point-in-polygon test for numpy arrays of *x* and *y*
	values, and a polygon described as 2-d numpy array (with shape (N,2))

	poly: N x 2 array

	Returns a numpy array of bools.
	'''
	x = np.atleast_1d(x)
	y = np.atleast_1d(y)
	inside = np.zeros(x.shape, bool)
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
	filt = np.zeros(x.shape, dtype=float)
	pinz = pi * x.flat[nz]
	filt.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
	filt[x == 0] = 1.
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
