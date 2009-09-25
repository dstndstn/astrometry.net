from astrometry.util.miscutils import *
from astrometry.util.pyfits_utils import *

# Produces a (cut-out of) the inverse-variance noise image, from columns
# [x0,x1] and rows [y0,y1] (inclusive).  Default is the whole image.
# "fpC" is a numpy array of the image (eg, pyfits.open(fpcfn)[0].data )
# "mask" is a pyfits object (eg, pyfits.open(maskfn) )
def sdss_noise_invvar(fpC, mask, x0=0, x1=None, y0=0, y1=None):
	if x1 is None:
		x1 = fpC.shape[1]-1
	if y1 is None:
		y1 = fpC.shape[0]-1

	# Poisson: mean = variance
	# Add readout noise?
	# Spatial smoothing?
	ivarimg = 1./fpC[y0:y1+1, x0:x1+1]
	# Noise model:
	#  -mask coordinates are wrt fpC coordinates.
	#  -INTERP, SATUR, CR,
	#  -GHOST?
	# HACK -- MAGIC -- these are the indices of INTER, SATUR, CR, and GHOST
	for i in [0, 1, 8, 9]:
		M = table_fields(mask[i+1].data)
		if M is None:
			continue
		for (c0,c1,r0,r1) in zip(M.cmin,M.cmax,M.rmin,M.rmax):
			(outx,nil) = get_overlapping_region(c0-x0, c1+1-x0, 0, x1-x0)
			(outy,nil) = get_overlapping_region(r0-y0, r1+1-y0, 0, y1-y0)
			ivarimg[outy,outx] = 0
	return ivarimg
