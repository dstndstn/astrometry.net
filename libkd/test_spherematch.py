# This file is part of libkd.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from astrometry.libkd import spherematch
import numpy as np
from time import time

N1 = 1000
N2 = 1000
D = 2
r = 0.02

x1 = np.random.rand(N1, D)
x2 = np.random.rand(N2, D)

t0 = time()
(inds,dists) = spherematch.match(x1, x2, r)
dt = time() - t0

print('spherematch.match: found', len(inds), 'pairs in', int(dt*1000.), 'ms')

order = np.argsort(inds[:,0]*N2 + inds[:,1])
inds = inds[order]
dists = dists[order]

t0 = time()
pairs = []
truedists = []
for i in range(N1):
    pt1 = x1[i,:]
    d2s = np.sum((x2 - pt1)**2, axis=1)
    good = np.where(d2s <= r**2)[0]
    for j in good:
        pairs.append((i, j))
        truedists.append(d2s[j])
dt = time() - t0
pairs = np.array(pairs)
truedists = np.sqrt(np.array(truedists))

print('naive            : found', len(pairs), 'pairs in', int(dt*1000.), 'ms')

order = np.argsort(pairs[:,0]*N2 + pairs[:,1])
pairs = pairs[order]

ok = np.array_equal(pairs, inds)
print('Indices equal:', ok)

ok = np.array_equal(truedists[order], dists.ravel())
print('Dists equal:', ok)


t0 = time()
(inds,dists) = spherematch.nearest(x1, x2, r)
dt = time() - t0

t0 = time()
inds = spherematch.match(x1, x2, r, indexlist=True)
dt = time() - t0

kd = spherematch.tree_build(x1)
spherematch.tree_print(kd)
kd.print()
R = spherematch.tree_search(kd, x2[0,:], 1.)
print('tree_search:', len(R), 'results')

