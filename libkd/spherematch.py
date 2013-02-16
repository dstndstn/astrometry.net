from astrometry.libkd import spherematch_c
import numpy as np

# for LSST (use things defined in astrometry.net 0.30)
try:
    from astrometry.util.starutil_numpy import radectoxyz, deg2dist, dist2deg, distsq2deg
except:
    from astrometry.util.starutil_numpy import radectoxyz, rad2distsq

    def rad2dist(r):
        return np.sqrt(rad2distsq(r))

    def distsq2rad(dist2):
        return np.arccos(1. - dist2 / 2.)
    def distsq2deg(dist2):
        return np.rad2deg(distsq2rad(dist2))

    # deg2dist, dist2deg
    def deg2dist(deg):
        return rad2dist(np.deg2rad(deg))
    def dist2deg(dist):
        return distsq2deg(dist**2)


# Copied from "celestial.py" by Sjoert van Velzen.
def match_radec(ra1, dec1, ra2, dec2, radius_in_deg, notself=False,
                nearest=False):
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

    if nearest:
        (inds,dists2) = _nearest_func(xyz2, xyz1, r, notself=notself)
        I = np.flatnonzero(inds >= 0)
        J = inds[I]
        d = distsq2deg(dists2[I])
    else:
        (inds,dists) = match(xyz1, xyz2, r, notself)
        dist_in_deg = dist2deg(dists)
        I,J = inds[:,0], inds[:,1]
        d = dist_in_deg[:,0]
        
    return (I, J, d)


def _cleaninputs(x1, x2):
    fx1 = x1.astype(np.float64)
    if x2 is x1:
        fx2 = fx1
    else:
        fx2 = x2.astype(np.float64)
    (N1,D1) = fx1.shape
    (N2,D2) = fx2.shape
    if D1 != D2:
        raise ValueError, 'Arrays must have the same dimensionality'
    return (fx1,fx2)

def _buildtrees(x1, x2):
    (fx1, fx2) = _cleaninputs(x1, x2)
    kd1 = spherematch_c.kdtree_build(fx1)
    if fx2 is fx1:
        kd2 = kd1
    else:
        kd2 = spherematch_c.kdtree_build(fx2)
    return (kd1, kd2)

def _freetrees(kd1, kd2):
    spherematch_c.kdtree_free(kd1)
    if kd2 != kd1:
        spherematch_c.kdtree_free(kd2)

def match(x1, x2, radius, notself=False):
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
    celestial positions; in particular, there is no RA wrapping at 0,
    and no special handling at the poles.  If you want to match
    celestial coordinates like RA,Dec, see the match_radec function.

    The "indices" return value has a row for each match; the matched
    points are:
    x1[indices[:,0],:]
    and
    x2[indices[:,1],:]

    This function doesn\'t know about spherical coordinates -- it just
    searches for matches in n-dimensional space.

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
    (kd1,kd2) = _buildtrees(x1, x2)
    #print 'spherematch.match: notself=', notself
    (inds,dists) = spherematch_c.match(kd1, kd2, radius, notself)
    _freetrees(kd1, kd2)
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

def nearest(x1, x2, maxradius, notself=False):
    '''
    For each point in x2, returns the index of the nearest point in x1,
    if there is a point within 'maxradius'.

    (Note, this may be backward from what you want/expect!)
    '''
    (kd1,kd2) = _buildtrees(x1, x2)
    (inds,dist2s) = spherematch_c.nearest(kd1, kd2, maxradius, notself)
    _freetrees(kd1, kd2)
    return (inds,dist2s)
_nearest_func = nearest

def tree_build_radec(ra=None, dec=None, xyz=None):
    if ra is not None:
        (N,) = ra.shape
        xyz = np.zeros((N,3)).astype(float)
        xyz[:,2] = np.sin(np.deg2rad(dec))
        cosd = np.cos(np.deg2rad(dec))
        xyz[:,0] = cosd * np.cos(np.deg2rad(ra))
        xyz[:,1] = cosd * np.sin(np.deg2rad(ra))
    kd = spherematch_c.kdtree_build(xyz)
    return kd

def tree_build(X):
    '''
    X: Numpy array of shape (N,D)
    Returns: kdtree identifier.
    '''
    return spherematch_c.kdtree_build(X)

def tree_free(kd):
    spherematch_c.kdtree_free(kd)

def tree_save(kd, fn):
    rtn = spherematch_c.kdtree_write(kd, fn)
    return rtn

def tree_open(fn):
    return spherematch_c.kdtree_open(fn)

def tree_close(kd):
    return spherematch_c.kdtree_close(kd)
    
def trees_match(kd1, kd2, radius, nearest=False, notself=False):
    '''
    Runs rangesearch or nearest-neighbour matching on given kdtrees.

    'radius' is Euclidean distance.

	If nearest=True, returns the nearest neighbour of each point in "kd1";
	ie, "I" will NOT contain duplicates, but "J" may.
	
    Returns (I, J, d), where
      I are indices into kd1
      J are indices into kd2
      d are distances-squared

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

	'''
    if nearest:
        J,I,d = spherematch_c.nearest2(kd2, kd1, radius, notself)
        d = distsq2deg(d)
    else:
        (inds,dists) = spherematch_c.match(kd1, kd2, radius)
        d = dist2deg(dists[:,0])
        I,J = inds[:,0], inds[:,1]
    return I,J,d

tree_bbox = spherematch_c.kdtree_bbox
tree_n = spherematch_c.kdtree_n

if __name__ == '__main__':
    import doctest
    doctest.testmod()
