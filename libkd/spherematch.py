import spherematch_c
import numpy

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

