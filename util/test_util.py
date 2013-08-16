from util import *
import numpy as np
import time

import fitsio
#X = fitsio.read('nasty.fits')
X = fitsio.read('dsky.fits')
assert(np.all(np.isfinite(X)))
print 'med', np.median(X)
print 'flat...'
f = flat_median_f(X)
print 'flat', f

for seed in range(42, 100):
    np.random.seed(seed)

    #X = np.random.normal(scale=10.0, size=(1016,1016)).astype(np.float32)
    X = np.random.normal(scale=10.0, size=(1015,1015)).astype(np.float32)
    # X = np.random.normal(scale=10.0, size=(10,10)).astype(np.float32)

    for i in range(3):
        t0 = time.clock()
        m = np.median(X)
        t1 = time.clock() - t0
        print 'np.median:', t1
    print 'value:', m

    I = np.argsort(X.ravel())
    m = X.flat[I[len(I)/2]]
    print 'element[N/2] =', m

    for i in range(3):
        t0 = time.clock()
        pym = flat_median_f(X)
        t1 = time.clock() - t0
        print 'flat_median:', t1
    print 'value:', pym
    assert(pym == m)
        
wcs = Tan()

wcs.crval = (1.,2.)
print 'crval', wcs.crval
(cr0,cr1) = wcs.crval

wcs.crpix = (50,100)
print 'crpix', wcs.crpix

wcs.crpix[0] = 500
print 'crpix', wcs.crpix

y = wcs.crpix[1]
wcs.crval[0] = 1.

wcs.cd = [1e-4,2e-4,-3e-4,4e-4]
print 'cd', wcs.cd

print 'wcs:', wcs

#wcs = tan_t()
wcs.pixel_scale()
xyz = wcs.pixelxy2xyz(0, 0)
print 'xyz', xyz
rd = wcs.pixelxy2radec(0, 0)
print 'rd', rd
xy = wcs.radec2pixelxy(rd[0], rd[1])
print 'xy', xy

X,Y = np.array([1,2,3]), np.array([4,5,6])
print 'X,Y', X,Y
R,D = wcs.pixelxy2radec(X, Y)
print 'R,D', R,D


wcs = Tan(0., 0., 0., 0., 1e-3, 0., 0., 1e-3, 100., 100.)

x1 = np.arange(3.)
x2 = np.arange(3)
x3 = x1[np.newaxis,:]
x4 = x1[:,np.newaxis]
x5 = x1.astype(np.float32)
x6 = x1.astype(np.dtype(np.float64).newbyteorder('S'))
x7 = 3.
x8 = np.array(3.)

import gc
gc.collect()
gc.set_debug(gc.DEBUG_LEAK)

for r,d in [(1., 2.),
            (1., x1),
            (1, x2),
            (1, x1),
            (x1, x1),
            (x1, x3),
            (x3, x4),
            (1., x5),
            (1., x6),
            (1., x7),
            (1., x8),
            ]:
    print
    print 'testing radec2pixelxy'
    print '  r', type(r),
    if hasattr(r, 'dtype'):
       print r.dtype,
    print r

    print '  d', type(d),
    if hasattr(d, 'dtype'):
        print d.dtype,
    print d
    
    ok,x,y = wcs.radec2pixelxy(r, d)
    print 'ok,x,y =', ok,x,y
    print '  x', type(x),
    if hasattr(x, 'dtype'):
        print x.dtype,
    print x
    print '  y', type(y),
    if hasattr(y, 'dtype'):
        print y.dtype,
    print y

    
gc.collect()

print 'Garbage:', len(gc.garbage)
for x in gc.garbage:
    print '  ', x
    
