import spherematch
from numpy import *
from numpy.random import rand
from time import time

N1 = 1000
N2 = 1000
D = 2
r = 0.02

x1 = rand(N1, D)
x2 = rand(N2, D)

t0 = time()
inds = spherematch.match(x1, x2, r)
dt = time() - t0

print 'C     : found', len(inds), 'pairs in', int(dt*1000.), 'ms'

order = argsort(inds[:,0]*N2 + inds[:,1])
inds = inds[order]

t0 = time()
pairs = []
for i in range(N1):
    pt1 = x1[i,:]
    d2s = sum((x2 - pt1)**2, axis=1)
    good = where(d2s <= r**2)[0]
    for j in good:
        pairs.append((i, j))
pairs = array(pairs)
dt = time() - t0

print 'Python: found', len(pairs), 'pairs in', int(dt*1000.), 'ms'

order = argsort(pairs[:,0]*N2 + pairs[:,1])
pairs = pairs[order]

#print pairs

ok = array_equal(pairs, inds)
print 'Equal:', ok

