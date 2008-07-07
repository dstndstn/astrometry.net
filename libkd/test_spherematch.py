import spherematch
from numpy import *
from numpy.random import rand
#from pylab import *

N1 = 100
N2 = 100
D = 2
r = 0.02

x1 = rand(N1, D)
x2 = rand(N2, D)

kd1 = spherematch.kdtree_build(x1)
kd2 = spherematch.kdtree_build(x2)

inds = spherematch.match(kd1, kd2, r)
print 'C: found', len(inds), 'pairs'

order = argsort(inds[:,0]*N2 + inds[:,1])
inds = inds[order]

spherematch.kdtree_free(kd1)
spherematch.kdtree_free(kd2)

pairs = []
for i in range(N1):
    pt1 = x1[i,:]
    d2s = sum((x2 - pt1)**2, axis=1)
    good = where(d2s <= r**2)[0]
    for j in good:
        pairs.append((i, j))
pairs = array(pairs)

print 'Python: found', len(pairs), 'pairs'

order = argsort(pairs[:,0]*N2 + pairs[:,1])
pairs = pairs[order]

#print pairs

ok = array_equal(pairs, inds)
print 'Equal:', ok

