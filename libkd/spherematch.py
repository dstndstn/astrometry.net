# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from astrometry.libkd import spherematch_c

from astrometry.util.starutil_numpy import radectoxyz, deg2dist, dist2deg, distsq2deg

import numpy as np

def match_xy(x1,y1, x2,y2, R, **kwargs):
    '''
    Like match_radec, except for plain old 2-D points.
    '''
    I,d = match(np.vstack((x1,y1)).T, np.vstack((x2,y2)).T, R, **kwargs)
    return I[:,0],I[:,1],d
    
# Copied from "celestial.py" by Sjoert van Velzen.
def match_radec(ra1, dec1, ra2, dec2, radius_in_deg, notself=False,
                nearest=False, indexlist=False, count=False):
    '''
    Cross-matches numpy arrays of RA,Dec points.

    Behaves like spherematch.pro of IDL.

    Parameters
    ----------
    ra1, dec1, ra2, dec2 : numpy arrays, or scalars.
        RA,Dec in degrees of points to match.

    radius_in_deg : float
        Search radius in degrees.

    notself : boolean
        If True, avoids returning 'identity' matches;
        ASSUMES that ra1,dec1 == ra2,dec2.

    nearest : boolean
        If True, returns only the nearest match in *(ra2,dec2)*
        for each point in *(ra1,dec1)*.

    indexlist : boolean
        If True, returns a list of length *len(ra1)*, containing *None*
        or a list of ints of matched points in *ra2,dec2*.


    Returns
    -------
    m1 : numpy array of integers
        Indices into the *ra1,dec1* arrays of matching points.
    m2 : numpy array of integers
        Same, but for *ra2,dec2*.
    d12 : numpy array, float
        Distance, in degrees, between the matching points.
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
            print('I', I.shape, I.dtype)
            print('J', J.shape, J.dtype)
            print('counts', counts.shape, counts.dtype)
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





def _cleaninputs(x1, x2):
    fx1 = x1.astype(np.float64)
    if x2 is x1:
        fx2 = fx1
    else:
        fx2 = x2.astype(np.float64)
    (N1,D1) = fx1.shape
    (N2,D2) = fx2.shape
    if D1 != D2:
        raise ValueError('Arrays must have the same dimensionality')
    return (fx1,fx2)

def _buildtrees(x1, x2):
    (fx1, fx2) = _cleaninputs(x1, x2)
    kd1 = spherematch_c.KdTree(fx1)
    if fx2 is fx1:
        kd2 = kd1
    else:
        kd2 = spherematch_c.KdTree(fx2)
    return (kd1, kd2)

def match(x1, x2, radius, notself=False, permuted=True, indexlist=False):
    '''
    ::

        (indices,dists) = match(x1, x2, radius):

    Or::

        inds = match(x1, x2, radius, indexlist=True):

    Returns the indices (Nx2 int array) and distances (Nx1 float
    array) between points in *x1* and *x2* that are within *radius*
    Euclidean distance of each other.

    *x1* is N1xD and *x2* is N2xD.  *x1* and *x2* can be the same
    array.  Dimensions D above 5-10 will probably not run faster than
    naive.

    Despite the name of this package, the arrays x1 and x2 need not be
    celestial positions; in particular, there is no RA wrapping at 0,
    and no special handling at the poles.  If you want to match
    celestial coordinates like RA,Dec, see the match_radec function.

    If *indexlist* is True, the return value is a python list with one
    element per data point in the first tree; that element is a python
    list containing the indices of points matched in the second tree.

    The *indices* return value has a row for each match; the matched
    points are:
    x1[indices[:,0],:]
    and
    x2[indices[:,1],:]

    This function doesn\'t know about spherical coordinates -- it just
    searches for matches in n-dimensional space.

    >>> from astrometry.util.starutil_numpy import *   
    >>> from astrometry.libkd import spherematch
    >>> # RA,Dec in degrees
    >>> ra1  = array([  0,  1, 2, 3, 4, 359,360])
    >>> dec1 = array([-90,-89,-1, 0, 1,  89, 90])
    >>> # xyz: N x 3 array: unit vectors
    >>> xyz1 = radectoxyz(ra1, dec1)
    >>> ra2  = array([ 45,   1,  4, 4, 4,  0,  1])
    >>> dec2 = array([-89, -88, -1, 0, 2, 89, 89])
    >>> xyz2 = radectoxyz(ra2, dec2)
    >>> # The \'radius\' is now distance between points on the unit sphere --
    >>> # for small angles, this is ~ angular distance in radians.  You can use
    >>> # the function:
    >>> radius_in_deg = 2.
    >>> r = sqrt(deg2distsq(radius_in_deg))
    >>> (inds,dists) = spherematch.match(xyz1, xyz2, r)
    >>> # Now *inds* is an Mx2 array of the matching indices,
    >>> # and *dists* the distances between them:
    >>> #  eg,  sqrt(sum((xyz1[inds[:,0],:] - xyz2[inds[:,1],:])**2, axis=1)) = dists
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

    Parameters
    ----------
    x1 : numpy array, float, shape N1 x D
        First array of points to match

    x2 : numpy array, float, shape N2 x D
        Second array of points to match

    radius : float
        Scalar Euclidean distance to match
        
    Returns
    -------
    indices : numpy array, integers, shape M x 2, for M matches
        The array of matching indices; *indices[:,0]* are indices in *x1*,
        *indices[:,1]* are indices in *x2*.

    dists : numpy array, floats, length M, for M matches
        The distances between matched points.
        
    If *indexlist* is *True*:

    indices : list of ints of integers
        The list of matching indices.  One list element per *x1* element,
        containing a list of matching indices in *x2*.
        
    '''
    (kd1,kd2) = _buildtrees(x1, x2)
    if indexlist:
        inds = spherematch_c.match2(kd1, kd2, radius, notself, permuted)
    else:
        (inds,dists) = spherematch_c.match(kd1, kd2, radius, notself, permuted)
    if indexlist:
        return inds
    return (inds,dists)

def match_naive(x1, x2, radius, notself=False):
    ''' Does the same thing as match(), but the straight-forward slow
    way.  (Not necessarily the way you\'d do it in python either).
    Not very fair as a speed comparison, but useful to convince
    yourself that match() does the right thing.
    '''
    (fx1, fx2) = _cleaninputs(x1, x2)
    (N1,D1) = x1.shape
    (N2,D2) = x2.shape
    inds = []
    dists = []
    for i1 in range(N1):
        for i2 in range(N2):
            if notself and i1 == i2:
                continue
            d2 = sum((x1[i1,:] - x2[i2,:])**2)
            if d2 < radius**2:
                inds.append((i1,i2))
                dists.append(sqrt(d2))
    inds = array(inds)
    dists = array(dists)
    return (inds,dists)

def nearest(x1, x2, maxradius, notself=False, count=False):
    '''
    For each point in x2, returns the index of the nearest point in x1,
    if there is a point within 'maxradius'.

    (Note, this may be backward from what you want/expect!)
    '''
    (kd1,kd2) = _buildtrees(x1, x2)
    if count:
        X = spherematch_c.nearest2(kd1, kd2, maxradius, notself, count)
    else:
        X = spherematch_c.nearest(kd1, kd2, maxradius, notself)
    return X
_nearest_func = nearest

def tree_build_radec(ra=None, dec=None, xyz=None):
    '''
    Builds a kd-tree given *RA,Dec* or unit-sphere *xyz* coordinates.
    '''
    if ra is not None:
        (N,) = ra.shape
        xyz = np.zeros((N,3)).astype(float)
        xyz[:,2] = np.sin(np.deg2rad(dec))
        cosd = np.cos(np.deg2rad(dec))
        xyz[:,0] = cosd * np.cos(np.deg2rad(ra))
        xyz[:,1] = cosd * np.sin(np.deg2rad(ra))
    kd = spherematch_c.KdTree(xyz)
    return kd

def tree_build(X, nleaf=16, bbox=True, split=False):
    '''
    Builds a kd-tree given a numpy array of Euclidean points.
    
    Parameters
    ----------
    X: numpy array of shape (N,D)
        The points to index.
        
    Returns
    -------
    kd: integer
        kd-tree identifier (address).
    '''
    return spherematch_c.KdTree(X, nleaf=nleaf, bbox=bbox, split=split)

def tree_free(kd):
    '''
    Frees a kd-tree previously created with *tree_build*.
    '''
    print('No need for tree_free')
    pass

def tree_save(kd, fn):
    '''
    Writes a kd-tree to the given filename.
    '''
    print('Deprecated tree_save()')
    return kd.write(fn)
#rtn = spherematch_c.kdtree_write(kd, fn)
#return rtn

def tree_open(fn, treename=None):
    '''
    Reads a kd-tree from the given filename.
    '''
    if treename is None:
        return spherematch_c.KdTree(fn)
    else:
        return spherematch_c.KdTree(fn, treename)

def tree_close(kd):
    '''
    Closes a kd-tree previously opened with *tree_open*.
    '''
    print('No need for tree_close')
    pass

def tree_search(kd, pos, radius, getdists=False, sortdists=False):
    '''
    Searches the given kd-tree for points within *radius* of the given
    position *pos*.
    '''
    #print('Unnecessary call to tree_search(kd, ...); use kd.search(...)')
    return kd.search(pos, radius, int(getdists), int(sortdists))

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

def trees_match(kd1, kd2, radius, nearest=False, notself=False,
                permuted=True, count=False):
    '''
    Runs rangesearch or nearest-neighbour matching on given kdtrees.

    'radius' is Euclidean distance.

    If 'nearest'=True, returns the nearest neighbour of each point in "kd1";
    ie, "I" will NOT contain duplicates, but "J" may.

    If 'count'=True, also counts the number of objects within range
    as well as returning the nearest neighbor of each point in "kd1";
    the return value becomes I,J,d,counts , counts a numpy array of ints.

    Returns (I, J, d), where
      I are indices into kd1
      J are indices into kd2
      d are distances-squared
      [counts is number of sources in range]

    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 6]]).T.astype(float)
    >>> Y = np.array([[1, 4, 4]]).T.astype(float)
    >>> kd1 = tree_build(X)
    >>> kd2 = tree_build(Y)
    >>> I,J,d = trees_match(kd1, kd2, 1.1, nearest=True)
    >>> print I
    [0 1 2]
    >>> print J
    [0 0 2]
    >>> print d
    [  0.  60.  60.]
    >>> I,J,d,count = trees_match(kd1, kd2, 1.1, nearest=True, count=True)
    >>> print I
    [0 1 2]
    >>> print J
    [0 0 2]
    >>> print d
    [  0.  60.  60.]
    >>> print count
    [1 1 2]
    '''
    rtn = None
    if nearest:
        rtn = spherematch_c.nearest2(kd2, kd1, radius, notself, count)
        # J,I,d,[count]
        rtn = (rtn[1], rtn[0], np.sqrt(rtn[2])) + rtn[3:]
        #distsq2deg(rtn[2]),
    else:
        (inds,dists) = spherematch_c.match(kd1, kd2, radius, notself, permuted)
        #d = dist2deg(dists[:,0])
        d = dists[:,0]
        I,J = inds[:,0], inds[:,1]
        rtn = (I,J,d)
    return rtn

def tree_permute(kd, I):
    print('Unnecessary call to tree_permute(kd, I): use kd.permute(I)')
    return kd.permute(I)

def tree_bbox(kd):
    print('Unnecessary call to tree_bbox(kd): use kd.bbox')
    return kd.bbox

def tree_n(kd):
    print('Unnecessary call to tree_n(kd): use kd.n')
    return kd.n

def tree_print(kd):
    print('Unnecessary call to tree_print(kd): use kd.print()')
    kd.print()
    
def tree_data(kd, I):
    print('Unnecessary call to tree_data(kd, I): use kd.get_data(I)')
    return kd.get_data(I)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
