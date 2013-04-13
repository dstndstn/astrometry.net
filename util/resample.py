import numpy as np
import scipy.interpolate as interp
from miscutils import lanczos_filter

def resample_with_wcs(targetwcs, wcs, Limages, L):
	'''
	Returns (Yo,Xo, Yi,Xi, ims)

	Limages: list of images to Lanczos-interpolate at the given Lanczos order.
	If empty, just returns nearest-neighbour indices.

	L: int, lanczos order

	Use the results like:

	target[Yo,Xo] = nearest_neighbour[Yi,Xi]

	target[Yo,Xo] = ims[i]
	
	'''
	# Adapted from detection/sdss-demo.py

	H,W = int(targetwcs.imageh), int(targetwcs.imagew)
	h,w = int(wcs.imageh), int(wcs.imagew)

	for im in Limages:
		assert(im.shape == (h,w))

	print 'Target size', W, H
	print 'Input size', w, h
	
	# First find the approximate bbox of the input image in
	# the target image so that we don't ask for way too
	# many out-of-bounds pixels...
	XY = []
	for x,y in [(0,0), (w-1,0), (w-1,h-1), (0, h-1)]:
		#rd = wcs.pixelToPosition(x,y)
		#XY.append(targetwcs.positionToPixel(rd))
		ra,dec = wcs.pixelxy2radec(float(x + 1), float(y + 1))
		ok,x,y = targetwcs.radec2pixelxy(ra, dec)
		XY.append((x - 1,y - 1))
	XY = np.array(XY)
	# Now we build a spline that maps "target" pixels to "input" pixels
	# spline inputs: pixel coords in the 'output' image
	x0,y0 = np.round(XY.min(axis=0)).astype(int)
	x1,y1 = np.round(XY.max(axis=0)).astype(int)
	margin = 20
	step = 25
	xx = np.arange(max(0, x0-margin), min(W, x1+margin+step), step)
	yy = np.arange(max(0, y0-margin), min(H, y1+margin+step), step)
	if (len(xx) == 0) or (len(yy) == 0):
		raise RuntimeError('No overlap between input and target WCSes')
	if (len(xx) <= 3) or (len(yy) <= 3):
		raise RuntimeError('Not enough overlap between input and target WCSes')

	XYo = []
	for y in yy:
		for x in xx:
			#rd = targetwcs.pixelToPosition(x,y)
			#XYo.append(wcs.positionToPixel(rd))
			ra,dec = targetwcs.pixelxy2radec(float(x + 1), float(y + 1))
			ok,xw,yw = wcs.radec2pixelxy(ra,dec)
			XYo.append((xw - 1, yw - 1))
	XYo = np.array(XYo)
	# spline outputs -- pixel coords in the 'input' image
	Xo = XYo[:,0].reshape(len(yy), len(xx))
	Yo = XYo[:,1].reshape(len(yy), len(xx))

	xspline = interp.RectBivariateSpline(xx, yy, Xo.T)
	yspline = interp.RectBivariateSpline(xx, yy, Yo.T)

	# Now, build the full pixel grid we want to interpolate...
	ixo = np.arange(max(0, x0-margin), min(W, x1+margin+1), dtype=int)
	iyo = np.arange(max(0, y0-margin), min(H, y1+margin+1), dtype=int)

	# And run the interpolator.  [xy]spline() does a meshgrid-like broadcast,
	# so fxi,fyi have shape n(iyo),n(ixo)
	fxi = xspline(ixo, iyo).T
	fyi = yspline(ixo, iyo).T

	print 'ixo', ixo.shape
	print 'iyo', iyo.shape
	print 'fxi', fxi.shape
	print 'fyi', fyi.shape

	ixi = np.round(fxi).astype(int)
	iyi = np.round(fyi).astype(int)

	# Keep only in-bounds pixels.
	I = np.flatnonzero((ixi >= 0) * (iyi >= 0) * (ixi < w) * (iyi < h))

	fxi = fxi[I]
	fyi = fyi[I]
	ixi = ixi[I]
	iyi = iyi[I]
	ixo = ixo[I]
	iyo = iyo[I]

	assert(np.all(ixi >= 0))
	assert(np.all(iyi >= 0))
	assert(np.all(ixi < w))
	assert(np.all(iyi < h))

	if len(Limages):
		fxi -= ixi
		fyi -= iyi
		dx = fxi
		dy = fyi
		del fxi
		del fyi

		# Lanczos interpolation.
		# number of pixels
		nn = len(ixo)
		NL = 2*L+1

		# We interpolate all the pixels at once.

		# accumulators for each input image
		laccs = [np.zeros(nn) for im in Limages]
		# sum of lanczos terms
		fsum = np.zeros(nn)

		off = np.arange(-L, L+1)
		for oy in off:
			fy = lanczos_filter(L, oy + dy)
			for ox in off:
				fx = lanczos_filter(L, ox + dx)
				for lacc,im in zip(laccs, Limages):
					lacc += fx * fy * im[np.clip(iyi + oy, 0, h-1),
										 np.clip(ixi + ox, 0, w-1)]
				fsum += fx*fy
		for lacc in laccs:
			lacc /= fsum

		rims = laccs

	else:
		rims = []

	return (ixo,iyo, ixi,iyi, rims)
