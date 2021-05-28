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


### Repeat for u64 trees

S = float(1<<63)
ux1 = (x1 * S).astype(np.uint64)
ux2 = (x2 * S).astype(np.uint64)

t0 = time()
(inds,dists) = spherematch.match(ux1, ux2, r*S)
dt = time() - t0

print('spherematch.match: found', len(inds), 'pairs in', int(dt*1000.), 'ms with uint64 trees')
order = np.argsort(inds[:,0]*N2 + inds[:,1])
inds = inds[order]
dists = dists[order]
ok = np.array_equal(pairs, inds)
print('Indices equal:', ok)

print('Build uint64 tree...')
kd = spherematch.tree_build(ux1)
kd.print()

data = kd.get_data(np.array([0,3,5]).astype(np.uint32))
assert(data.dtype == np.uint64)
print('Kd data:', data.dtype, data)

kd.write('kd-u64.fits')

kd2 = spherematch.tree_open('kd-u64.fits')
data2 = kd2.get_data(np.array([0,3,5]).astype(np.uint32))
assert(data2.dtype == np.uint64)
print('Kd data2:', data2.dtype, data2)
assert(np.all(data == data2))

del kd
del kd2

###



t0 = time()
(inds,dists) = spherematch.nearest(x1, x2, r)
dt = time() - t0

t0 = time()
inds = spherematch.match(x1, x2, r, indexlist=True)
dt = time() - t0

kd = spherematch.tree_build(x1)
kd.print()
R = spherematch.tree_search(kd, x2[0,:], 1.)
print('tree_search:', len(R), 'results')
spherematch.tree_close(kd)

kd2 = spherematch.tree_build(x2)
I,J,d = spherematch.trees_match(kd, kd2, 1.)
print('trees_match:', len(I), 'matches')

I,J,d = spherematch.trees_match(kd, kd2, 1., nearest=True)
print('trees_match:', len(I), 'matches (nearest)')

print('Kd bounding-box:', spherematch.tree_bbox(kd))

print('Kd bounding-box:', kd.bbox)

print('Kd data:', spherematch.tree_data(kd, np.array([0,3,5]).astype(np.uint32)))

print('Kd data:', kd.get_data(np.array([0,3,5]).astype(np.uint32)))

print('Permute:', spherematch.tree_permute(kd, np.array([3,5,7]).astype(np.int32)))

print('Permute:', kd.permute(np.array([0,99,199]).astype(np.int32)))

ra,dec = np.meshgrid(np.arange(0, 360), np.arange(-90, 91, 1))
ra1 = ra.ravel()
dec1 = dec.ravel()
rdkd1 = spherematch.tree_build_radec(ra1, dec1)
print('RdKd:', rdkd1.n, rdkd1.bbox)

ra2  = np.random.uniform(-10, 10, size=1000)
dec2 = np.random.uniform(-10, 10, size=1000)
rdkd2 = spherematch.tree_build_radec(ra2, dec2)

I = spherematch.tree_search_radec(rdkd1, ra2[0], dec2[0], 2.)
print('search_radec:', I)

I,J,d = spherematch.match_radec(ra1, dec1, ra2, dec2, 1.)
print('Matches:', len(I))

I,J,d = spherematch.match_radec(ra1, dec1, ra2, dec2, 1., nearest=True)
print('Nearest matches:', len(I))

I = spherematch.match_radec(ra1, dec1, ra2, dec2, 1.,
                            indexlist=True)
print('Index lists matches:', len(I))
