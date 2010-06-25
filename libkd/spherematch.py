import spherematch_c
from math import *
from numpy import *

'''
match(x1, x2, radius):



'''
def match(x1, x2, radius):
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
