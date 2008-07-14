import spherematch_c
from math import *
from numpy import *

def match(x1, x2, radius):
    (N1,D1) = x1.shape
    (N2,D2) = x2.shape
    if D1 != D2:
        raise ValueError, 'Arrays must have the same dimensionality'
    kd1 = spherematch_c.kdtree_build(x1)
    kd2 = spherematch_c.kdtree_build(x2)
    inds = spherematch_c.match(kd1, kd2, radius)
    spherematch_c.kdtree_free(kd1)
    spherematch_c.kdtree_free(kd2)
    return inds

def tree_build(ra=None, dec=None, xyz=None):
    if ra is not None:
        (N,) = ra.shape
        xyz = zeros((N,3)).astype(float)
        xyz[:,2] = sin(radians(dec))
        cosd = cos(radians(dec))
        xyz[:,0] = cosd * cos(radians(ra))
        xyz[:,1] = cosd * sin(radians(ra))
    kd = spherematch_c.kdtree_build(xyz)
    return kd

def tree_save(kd, fn):
    rtn = spherematch_c.kdtree_write(kd, fn)
    return rtn

