from util import *


wcs = Tan()
#wcs = tan_t()
wcs.pixel_scale()
xyz = wcs.pixelxy2xyz(0, 0)
