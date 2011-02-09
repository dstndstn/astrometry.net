from util import *
import numpy as np

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
