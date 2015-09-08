# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
try:
    import pyfits
except ImportError:
    try:
        from astropy.io import fits as pyfits
    except ImportError:
        raise ImportError("Cannot import either pyfits or astropy.io.fits")
import math
from math import exp
from matplotlib.pylab import imread
from numpy.oldnumeric.functions import zeros, ravel

I=imread('3.png')
I=I[:,:,:3]
(h,w,planes) = I.shape
XY = pyfits.open('16b.fits')[1].data
X = XY.field('X')
Y = XY.field('Y')

psfw = 1.0
stars = zeros((h,w)).astype(float)

for (x,y) in zip(X,Y):
    ix = int(round(x))
    iy = int(round(y))
    for dy in range(-5, 6):
        yy = iy + dy
        if yy < 0 or yy >= h:
            continue
        for dx in range(-5, 6):
            xx = ix + dx
            if xx < 0 or xx >= w:
                continue
            dd = (xx - x)**2 + (yy - y)**2
            stars[yy,xx] += exp(-dd / (2 * psfw**2)) #1./(psfw**2 * 2 * math.pi

#origfrac = 0.5
#maxorig = I.max()
#starfrac = (1.0 - origfrac) + (1.0 - maxorig)
#for p in range(planes):
#    I[:,:,p] = I[:,:,p] * origfrac + stars/stars.max() * starfrac

for p in range(planes):
    I[:,:,p] = I[:,:,p] * 0.7 + stars/stars.max() * 0.8

f=open('out.ppm', 'wb')
f.write('P6 %i %i %i\n' % (w, h, 255))
#for j in range(h):
#    for i in range(w):
#        for p in range(planes):
#            f.write(chr(int(round(I[j,i,p] * 255.0))))
flatI = (I.ravel() * 255.0).round().astype(int)
f.write("".join([chr(min(i,255)) for i in flatI]))
f.close()
