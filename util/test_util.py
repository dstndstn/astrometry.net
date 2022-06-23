# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import time
from .util import *
from .starutil_numpy import *


tan = Tan(5., 10., 500.5, 500.5,
          1e-3, 1e-6, 1e-7, 1e-3, 1000., 1000.)

r1,d1 = tan.pixelxy2radec(0.5, 0.5)
r2,d2 = tan.pixelxy2radec(1000.5, 0.5)
r3,d3 = tan.pixelxy2radec(0.5, 1000.5)
r4,d4 = tan.pixelxy2radec(1000.5, 1000.5)
r5,d5 = tan.pixelxy2radec(500.5, 500.5)

print('1', r1,d1)
print('2', r2,d2)
print('3', r3,d3)
print('4', r4,d4)
print('5', r5,d5)

halftan = tan.scale(0.5)

print('Half TAN:', halftan)

ok,x1,y1 = halftan.radec2pixelxy(r1, d1)
ok,x2,y2 = halftan.radec2pixelxy(r2, d2)
ok,x3,y3 = halftan.radec2pixelxy(r3, d3)
ok,x4,y4 = halftan.radec2pixelxy(r4, d4)
ok,x5,y5 = halftan.radec2pixelxy(r5, d5)

print('half1', x1,y1)
print('half2', x2,y2)
print('half3', x3,y3)
print('half4', x4,y4)
print('half5', x5,y5)

print('------')

sip = Sip(Tan(5., 10., 500.5, 500.5,
              1e-3, 1e-6, 1e-7, 1e-3, 1000., 1000.))
sip.set_a_term(2, 0, 1e-5)
sip.set_a_term(2, 1, -2e-8)
sip.set_b_term(2, 1, 1e-8)
sip.set_b_term(0, 3, 3e-8)
sip.a_order = 2
sip.b_order = 4

print('Sip:', sip)

sip_ensure_inverse_polynomials(sip)

print('Sip:', sip)

print('A:')
print(np.array(sip.a).reshape((10,10))[:4,:4])
print('B:')
print(np.array(sip.b).reshape((10,10))[:4,:4])
print('AP:')
print(np.array(sip.ap).reshape((10,10))[:5,:5])
print('BP:')
print(np.array(sip.bp).reshape((10,10))[:5,:5])

from .plotSipDistortion import plotDistortion
import pylab as plt

plotDistortion(sip, 1000, 1000, 15, exaggerate=10.)
plt.savefig('sip1.png')

r1,d1 = sip.pixelxy2radec(0.5, 0.5)
r2,d2 = sip.pixelxy2radec(1000.5, 0.5)
r3,d3 = sip.pixelxy2radec(0.5, 1000.5)
r4,d4 = sip.pixelxy2radec(1000.5, 1000.5)
r5,d5 = sip.pixelxy2radec(500.5, 500.5)

xx = np.linspace(0.5, 1000.5, 100)
yy = [0.5, 500.5, 1000.5]
for iy,y in enumerate(yy):
    XD,YD,XU,YU = [],[],[],[]
    for x in xx:
        xd,yd = sip.get_distortion(x, y)
        xu,yu = sip.get_undistortion(xd, yd)
        XD.append(xd)
        YD.append(yd)
        XU.append(xu)
        YU.append(yu)
    XD = np.array(XD)
    YD = np.array(YD)
    XU = np.array(XU)
    YU = np.array(YU)

    plt.clf()
    plt.plot(xx, XD-xx, 'k-')
    plt.plot(xx, YD-y , 'k--')
    plt.plot(xx, XU-XD, 'r-')
    plt.plot(xx, YU-YD , 'r--')
    ex = 10.
    plt.plot(xx, ex * (XU - xx), 'b-')
    plt.plot(xx, ex * (YU - y), 'b--')
    plt.savefig('sip-dist%i.png' % iy)

print('1', r1,d1)
print('2', r2,d2)
print('3', r3,d3)
print('4', r4,d4)
print('5', r5,d5)

ok,x1,y1 = sip.radec2pixelxy(r1, d1)
ok,x2,y2 = sip.radec2pixelxy(r2, d2)
ok,x3,y3 = sip.radec2pixelxy(r3, d3)
ok,x4,y4 = sip.radec2pixelxy(r4, d4)
ok,x5,y5 = sip.radec2pixelxy(r5, d5)

print('inv1', x1,y1)
print('inv2', x2,y2)
print('inv3', x3,y3)
print('inv4', x4,y4)
print('inv5', x5,y5)

halfsip = sip.scale(0.5)

print('Half SIP:', halfsip)

print('A:')
print(np.array(halfsip.a).reshape((10,10))[:4,:4])
print('B:')
print(np.array(halfsip.b).reshape((10,10))[:4,:4])
print('AP:')
print(np.array(halfsip.ap).reshape((10,10))[:5,:5])
print('BP:')
print(np.array(halfsip.bp).reshape((10,10))[:5,:5])

plotDistortion(halfsip, 500, 500, 15, exaggerate=10.)
plt.savefig('sip2.png')

ok,x1,y1 = halfsip.radec2pixelxy(r1, d1)
ok,x2,y2 = halfsip.radec2pixelxy(r2, d2)
ok,x3,y3 = halfsip.radec2pixelxy(r3, d3)
ok,x4,y4 = halfsip.radec2pixelxy(r4, d4)
ok,x5,y5 = halfsip.radec2pixelxy(r5, d5)

print('half1', x1,y1)
print('half2', x2,y2)
print('half3', x3,y3)
print('half4', x4,y4)
print('half5', x5,y5)

rh1,dh1 = halfsip.pixelxy2radec(0.5, 0.5)
rh2,dh2 = halfsip.pixelxy2radec(500.5, 0.5)
rh3,dh3 = halfsip.pixelxy2radec(0.5, 500.5)
rh4,dh4 = halfsip.pixelxy2radec(500.5, 500.5)
rh5,dh5 = halfsip.pixelxy2radec(250.5, 250.5)

print('half1', rh1,dh1)
print('half2', rh2,dh2)
print('half3', rh3,dh3)
print('half4', rh4,dh4)
print('half5', rh5,dh5)

ok,x1,y1 = halfsip.radec2pixelxy(rh1, dh1)
ok,x2,y2 = halfsip.radec2pixelxy(rh2, dh2)
ok,x3,y3 = halfsip.radec2pixelxy(rh3, dh3)
ok,x4,y4 = halfsip.radec2pixelxy(rh4, dh4)
ok,x5,y5 = halfsip.radec2pixelxy(rh5, dh5)

print('hinv1', x1,y1)
print('hinv2', x2,y2)
print('hinv3', x3,y3)
print('hinv4', x4,y4)
print('hinv5', x5,y5)


import sys
sys.exit(0)



x = np.random.uniform(size=10000) * 10
y = np.random.uniform(size=10000) * 100
xlo,xhi = 0., 9.
ylo,yhi = 5., 105.
nx,ny = 10,12
H = np.zeros((ny,nx), np.int32)
an_hist2d(x, y, H, xlo, xhi, ylo, yhi)

print(H.sum())

H2,xe,ye = np.histogram2d(x, y, range=((xlo,xhi),(ylo,yhi)), bins=(nx,ny))

assert(np.all(H == H2.T))

x2 = np.append(x, np.array([xlo,xlo,xhi,xhi]))
y2 = np.append(y, np.array([ylo,yhi,ylo,yhi]))

Hb = np.zeros((ny,nx), np.int32)
an_hist2d(x2, y2, Hb, xlo, xhi, ylo, yhi)

H2b,xe,ye = np.histogram2d(x2, y2, range=((xlo,xhi),(ylo,yhi)), bins=(nx,ny))

assert(np.all(Hb == H2b.T))

an_hist2d(x, y, Hb, xlo, xhi, ylo, yhi)

assert(np.all(Hb == (H2b.T + H2.T)))


import sys
sys.exit(0)




sip = Sip('dec095705.01.p.w.wcs')
x = np.random.uniform(2000, size=100)
y = np.random.uniform(4000, size=100)
ra,dec = sip.pixelxy2radec(x, y)

xx = x + np.random.normal(scale=0.1, size=x.shape)
yy = y + np.random.normal(scale=0.1, size=y.shape)

xyz = radectoxyz(ra, dec)
xy = np.vstack((xx,yy)).T
print('xyz', xyz.shape)
print('xy', xy.shape)

tan = Tan('dec095705.01.p.w.wcs')

sip2 = fit_sip_wcs_2(xyz, xy, None, tan, 2,2)

print('Got', sip2)
print('Vs truth', sip)

sip2.write_to('sip2.wcs')

xy = np.vstack((x,y)).T
sip3 = fit_sip_wcs_2(xyz, xy, None, tan, 2,2)
sip3.write_to('sip3.wcs')

import sys
sys.exit(0)


X = np.random.uniform(1000., size=1001).astype(np.float32)

med = flat_median_f(X)
print('Median', med)
print('vs np ', np.median(X))

for pct in [0., 1., 10., 25., 50., 75., 99., 100.]:
    p1 = flat_percentile_f(X, pct)
    p2 = np.percentile(X, pct)
    print('Percentile', pct)
    print('  ', p1)
    print('vs', p2)


import fitsio
#X = fitsio.read('nasty.fits')
X = fitsio.read('dsky.fits')
assert(np.all(np.isfinite(X)))
print('med', np.median(X))
print('flat...')
f = flat_median_f(X)
print('flat', f)

for seed in range(42, 100):
    np.random.seed(seed)

    #X = np.random.normal(scale=10.0, size=(1016,1016)).astype(np.float32)
    X = np.random.normal(scale=10.0, size=(1015,1015)).astype(np.float32)
    # X = np.random.normal(scale=10.0, size=(10,10)).astype(np.float32)

    for i in range(3):
        t0 = time.process_time()
        m = np.median(X)
        t1 = time.process_time() - t0
        print('np.median:', t1)
    print('value:', m)

    I = np.argsort(X.ravel())
    m = X.flat[I[len(I)/2]]
    print('element[N/2] =', m)

    for i in range(3):
        t0 = time.process_time()
        pym = flat_median_f(X)
        t1 = time.process_time() - t0
        print('flat_median:', t1)
    print('value:', pym)
    assert(pym == m)
        
wcs = Tan()

wcs.crval = (1.,2.)
print('crval', wcs.crval)
(cr0,cr1) = wcs.crval

wcs.crpix = (50,100)
print('crpix', wcs.crpix)

wcs.crpix[0] = 500
print('crpix', wcs.crpix)

y = wcs.crpix[1]
wcs.crval[0] = 1.

wcs.cd = [1e-4,2e-4,-3e-4,4e-4]
print('cd', wcs.cd)

print('wcs:', wcs)

#wcs = tan_t()
wcs.pixel_scale()
xyz = wcs.pixelxy2xyz(0, 0)
print('xyz', xyz)
rd = wcs.pixelxy2radec(0, 0)
print('rd', rd)
xy = wcs.radec2pixelxy(rd[0], rd[1])
print('xy', xy)

X,Y = np.array([1,2,3]), np.array([4,5,6])
print('X,Y', X,Y)
R,D = wcs.pixelxy2radec(X, Y)
print('R,D', R,D)


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
    print()
    print('testing radec2pixelxy')
    print('  r', type(r), end=' ')
    if hasattr(r, 'dtype'):
       print(r.dtype, end=' ')
    print(r)

    print('  d', type(d), end=' ')
    if hasattr(d, 'dtype'):
        print(d.dtype, end=' ')
    print(d)
    
    ok,x,y = wcs.radec2pixelxy(r, d)
    print('ok,x,y =', ok,x,y)
    print('  x', type(x), end=' ')
    if hasattr(x, 'dtype'):
        print(x.dtype, end=' ')
    print(x)
    print('  y', type(y), end=' ')
    if hasattr(y, 'dtype'):
        print(y.dtype, end=' ')
    print(y)

    
gc.collect()

print('Garbage:', len(gc.garbage))
for x in gc.garbage:
    print('  ', x)
    
