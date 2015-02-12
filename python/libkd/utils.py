import numpy as np

# some helpers used in the sphere module

# RA, Dec in degrees: scalars or 1-d arrays.
# returns xyz of shape (N,3)
def radectoxyz(ra_deg, dec_deg):
	ra	= np.deg2rad(ra_deg)
	dec = np.deg2rad(dec_deg)
	cosd = np.cos(dec)
	xyz = np.vstack((cosd * np.cos(ra),
				  cosd * np.sin(ra),
				  np.sin(dec))).T
	assert(xyz.shape[1] == 3)
	return xyz
	
def deg2dist(deg):
	return rad2dist(np.deg2rad(deg))
	
def rad2dist(r):
	return np.sqrt(rad2distsq(r))
	
def rad2distsq(r):
	# inverse of distsq2arc; cosine law.
	return 2.0 * (1.0 - np.cos(r))
	
def dist2deg(dist):
	return distsq2deg(dist**2)
	
def distsq2deg(dist2):
	return np.rad2deg(distsq2rad(dist2))
	
def distsq2rad(dist2):
	return np.arccos(1. - dist2 / 2.)
