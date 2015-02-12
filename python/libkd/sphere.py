from __future__ import absolute_import

import numpy as np

from libkd import _libkd
from libkd.kd import *
from libkd.utils import *

# Copied from "celestial.py" by Sjoert van Velzen.
def match_radec(ra1, dec1, ra2, dec2, radius_in_deg, notself=False,
                nearest=False, indexlist=False, count=False):
    '''
    (m1,m2,d12) = match_radec(ra1,dec1, ra2,dec2, radius_in_deg)

    Cross-matches numpy arrays of RA,Dec points.

    Behaves like spherematch.pro of IDL 

    ra1,dec1 (and 2): RA,Dec in degrees of points to match.
       Must be scalars or numpy arrays.

       radius_in_deg: search radius in degrees.

    notself: if True, avoids returning 'identity' matches;
        ASSUMES that ra1,dec1 == ra2,dec2.

    nearest: if True, returns only the nearest match in (ra2,dec2)
        for each point in (ra1,dec1).

    indexlist: returns a list of length len(ra1), containing None or a
    list of ints of matched points in ra2,dec2.  Returns this list.
        
    Returns:

    m1: indices into the "ra1,dec1" arrays of matching points.
       Numpy array of ints.
    m2: same, but for "ra2,dec2".
    d12: distance, in degrees, between the matching points.
    '''

    # Convert to coordinates on the unit sphere
    xyz1 = radectoxyz(ra1, dec1)
    #if all(ra1 == ra2) and all(dec1 == dec2):
    if ra1 is ra2 and dec1 is dec2:
        xyz2 = xyz1
    else:
        xyz2 = radectoxyz(ra2, dec2)
    r = deg2dist(radius_in_deg)

    extra = ()
    if nearest:
        X = _nearest_func(xyz2, xyz1, r, notself=notself, count=count)
        if not count:
            (inds,dists2) = X
            I = np.flatnonzero(inds >= 0)
            J = inds[I]
            d = distsq2deg(dists2[I])
        else:
            #print 'X', X
            #(inds,dists2,counts) = X
            J,I,d,counts = X
            extra = (counts,)
            print 'I', I.shape, I.dtype
            print 'J', J.shape, J.dtype
            print 'counts', counts.shape, counts.dtype
    else:
        X = match(xyz1, xyz2, r, notself=notself, indexlist=indexlist)
        if indexlist:
            return X
        (inds,dists) = X
        dist_in_deg = dist2deg(dists)
        I,J = inds[:,0], inds[:,1]
        d = dist_in_deg[:,0]
        
    return (I, J, d) + extra


def cluster_radec(ra, dec, R, singles=False):
	'''
	Finds connected groups of objects in RA,Dec space.

	Returns a list of lists of indices that are connected,
	EXCLUDING singletons.

    If *singles* is *True*, also returns the indices of singletons.
	'''
	I,J,d = match_radec(ra, dec, ra, dec, R, notself=True)

	# 'mgroups' maps each index in a group to a list of the group members
	mgroups = {}
	# 'ugroups' is a list of the unique groups
	ugroups = []
	
	for i,j in zip(I,J):
		# Are both sources already in groups?
		if i in mgroups and j in mgroups:
			# Are they already in the same group?
			if mgroups[i] == mgroups[j]:
				continue
			# merge if they are different;
			# assert(they are disjoint)
			lsti = mgroups[i]
			lstj = mgroups[j]
			merge = lsti + lstj
			for k in merge:
				mgroups[k] = merge

			ugroups.remove(lsti)
			ugroups.remove(lstj)
			ugroups.append(merge)

		elif i in mgroups:
			# Add j to i's group
			lst = mgroups[i]
			lst.append(j)
			mgroups[j] = lst
		elif j in mgroups:
			# Add i to j's group
			lst = mgroups[j]
			lst.append(i)
			mgroups[i] = lst
		else:
			# Create a new group
			lst = [i,j]
			mgroups[i] = lst
			mgroups[j] = lst

			ugroups.append(lst)


	if singles:
		S = np.ones(len(ra), bool)
		for g in ugroups:
			S[np.array(g)] = False
		S = np.flatnonzero(S)
		return ugroups,S

	return ugroups

def tree_build_radec(ra=None, dec=None, xyz=None):
    if ra is not None:
        (N,) = ra.shape
        xyz = np.zeros((N,3)).astype(float)
        xyz[:,2] = np.sin(np.deg2rad(dec))
        cosd = np.cos(np.deg2rad(dec))
        xyz[:,0] = cosd * np.cos(np.deg2rad(ra))
        xyz[:,1] = cosd * np.sin(np.deg2rad(ra))
    kd = _libkd.kdtree_build(xyz)
    return kd

def tree_search_radec(kd, ra, dec, radius, getdists=False, sortdists=False):
    '''
    ra,dec in degrees
    radius in degrees
    '''
    dec = np.deg2rad(dec)
    cosd = np.cos(dec)
    ra = np.deg2rad(ra)
    pos = np.array([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)])
    rad = deg2dist(radius)
    return tree_search(kd, pos, rad, getdists=getdists, sortdists=sortdists)


