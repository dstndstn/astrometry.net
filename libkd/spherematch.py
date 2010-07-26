import spherematch_c
from math import *
from numpy import *

def match(x1, x2, radius):
	'''
	(indices,dists) = match(x1, x2, radius):

    Given an N1 x D1 array x1,
    and   an N2 x D2 array x2,
    and   radius:

	Returns the indices (Nx2 int array) and distances (Nx1 float
	array) between points in "x1" and "x2" that are within "radius"
	Euclidean distance of each other.

	"x1" is N1xD and "x2" is N2xD.  "x1" and "x2" can be the same
	array.  Dimensions D above 5-10 will probably not run faster than
	naive.

    Despite the name of this package, the arrays x1 and x2 need not be
    celestial positions; in particular, there is no RA wrapping at 0!

	The "indices" return value has a row for each match; the matched
	points are:
	x1[indices[:,0],:]
	and
	x2[indices[:,1],:]

	This function doesn\'t know about spherical coordinates -- it just
	searches for matches in n-dimensional space.  For RA,Dec arrays,
	convert them to positions on the unit sphere, eg via the function
	radectoxyz() in astrometry.util.starutil_numpy:

	>>> from astrometry.util.starutil_numpy import *   
	>>> from astrometry.libkd import spherematch

	# RA,Dec in degrees
	>>> ra1  = array([  0,  1, 2, 3, 4, 359,360])
	>>> dec1 = array([-90,-89,-1, 0, 1,  89, 90])

	# xyz: N x 3 array: unit vectors
	>>> xyz1 = radectoxyz(ra1, dec1)

	>>> ra2  = array([ 45,   1,  4, 4, 4,  0,  1])
	>>> dec2 = array([-89, -88, -1, 0, 2, 89, 89])
	>>> xyz2 = radectoxyz(ra2, dec2)

	# The \'radius\' is now distance between points on the unit sphere --
	# for small angles, this is ~ angular distance in radians.  You can use
	# the function:
	>>> radius_in_deg = 2.
	>>> r = sqrt(deg2distsq(radius_in_deg))
 
	>>> (inds,dists) = spherematch.match(xyz1, xyz2, r)

	# Now "inds" is an Mx2 array of the matching indices,
	# and "dists" the distances between them:
	#  eg,  sqrt(sum((xyz1[inds[:,0],:] - xyz2[inds[:,1],:])**2, axis=1)) = dists

	>>> print inds
	[[0 0]
	 [1 0]
	 [1 1]
	 [2 2]
	 [3 2]
	 [3 3]
	 [4 3]
	 [4 4]
	 [5 5]
	 [6 5]
	 [5 6]
	 [6 6]]
 
	>>> print sqrt(sum((xyz1[inds[:,0],:] - xyz2[inds[:,1],:])**2, axis=1))
	[ 0.01745307  0.01307557  0.01745307  0.0348995   0.02468143  0.01745307
	  0.01745307  0.01745307  0.0003046   0.01745307  0.00060917  0.01745307]

	>>> print dists[:,0]
	[ 0.01745307  0.01307557  0.01745307  0.0348995   0.02468143  0.01745307
	  0.01745307  0.01745307  0.0003046   0.01745307  0.00060917  0.01745307]

	>>> print vstack((ra1[inds[:,0]], dec1[inds[:,0]], ra2[inds[:,1]], dec2[inds[:,1]])).T
	[[  0 -90  45 -89]
	 [  1 -89  45 -89]
	 [  1 -89   1 -88]
	 [  2  -1   4  -1]
	 [  3   0   4  -1]
	 [  3   0   4   0]
	 [  4   1   4   0]
	 [  4   1   4   2]
	 [359  89   0  89]
	 [360  90   0  89]
	 [359  89   1  89]
	 [360  90   1  89]]

	'''
	x1 = x1.astype(float64)
	x2 = x2.astype(float64)
	(N1,D1) = x1.shape
	(N2,D2) = x2.shape
	if D1 != D2:
		raise ValueError, 'Arrays must have the same dimensionality'
	kd1 = spherematch_c.kdtree_build(x1)
	#if x1 is x2:
	kd2 = spherematch_c.kdtree_build(x2)
	(inds,dists) = spherematch_c.match(kd1, kd2, radius)
	spherematch_c.kdtree_free(kd1)
	spherematch_c.kdtree_free(kd2)
	return (inds,dists)

def match_naive(x1, x2, radius):
	(N1,D1) = x1.shape
	(N2,D2) = x2.shape
	if D1 != D2:
		raise ValueError, 'Arrays must have the same dimensionality'
	inds = []
	dists = []
	for i1 in range(N1):
		for i2 in range(N2):
			d2 = sum((x1[i1,:] - x2[i2,:])**2)
			if d2 < radius**2:
				inds.append((i1,i2))
				dists.append(sqrt(d2))
	inds = array(inds)
	dists = array(dists)
	return (inds,dists)

def nearest(x1, x2, maxradius):
	(N1,D1) = x1.shape
	(N2,D2) = x2.shape
	if D1 != D2:
		raise ValueError, 'Arrays must have the same dimensionality'
	kd1 = spherematch_c.kdtree_build(x1)
	kd2 = spherematch_c.kdtree_build(x2)
	(inds,dist2s) = spherematch_c.nearest(kd1, kd2, maxradius)
	spherematch_c.kdtree_free(kd1)
	spherematch_c.kdtree_free(kd2)
	return (inds,dist2s)

def rad2deg(x):
	return x * 180./pi

def deg2rad(x):
	return x * pi / 180

def tree_build(ra=None, dec=None, xyz=None):
	if ra is not None:
		(N,) = ra.shape
		print 'dec shape', dec.shape
		xyz = zeros((N,3)).astype(float)
		xyz[:,2] = sin(deg2rad(dec))
		cosd = cos(deg2rad(dec))
		xyz[:,0] = cosd * cos(deg2rad(ra))
		xyz[:,1] = cosd * sin(deg2rad(ra))
	kd = spherematch_c.kdtree_build(xyz)
	return kd

def tree_save(kd, fn):
	rtn = spherematch_c.kdtree_write(kd, fn)
	return rtn

def tree_open(fn):
	return spherematch_c.kdtree_open(fn)

def tree_close(kd):
	return spherematch_c.kdtree_close(kd)
	
def trees_match(kd1, kd2, radius):
	(inds,dists) = spherematch_c.match(kd1, kd2, radius)
	return (inds,dists)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
