import scipy.interpolate as interp

def resample_with_wcs(targetwcs, wcs, Limages, Lorder):
	'''
	Returns (Yo,Xo, Yi,Xi, ims)

	Limages: list of images to Lanczos-interpolate at the given Lanczos order.
	If empty, just returns nearest-neighbour indices.

	Use the results like:

	target[Yo,Xo] = nearest_neighbour[Yi,Xi]

	target[Yo,Xo] = ims[i]
	
	'''
	# Adapted from detection/sdss-demo.py

	H,W = targetwcs.imageh, targetwcs.imagew
	h,w = wcs.imageh, wcs.imagew
	
	# First find the approximate bbox of the input image in
	# the target image so that we don't ask for way too
	# many out-of-bounds pixels...
	XY = []
	for x,y in [(0,0), (w-1,0), (w-1,h-1), (0, h-1)]:
		#rd = wcs.pixelToPosition(x,y)
		#XY.append(targetwcs.positionToPixel(rd))
		ra,dec = wcs.pixelxy2radec(float(x), float(y))
		ok,x,y = targetwcs.radec2pixelxy(ra, dec)
		XY.append((x,y))
	XY = np.array(XY)
	# Now we build a spline that maps "target" pixels to "input" pixels
	# spline inputs:
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
			ra,dec = targetwcs.pixelxy2radec(float(x), float(y))
			ok,x,y = wcs.radec2pixelxy(ra,dec)
			XYo.append((x,y))
	XYo = np.array(XYo)
	# spline outputs -- pixel coords in the 'detmap'
	Xo = XYo[:,0].reshape(len(yy), len(xx))
	Yo = XYo[:,1].reshape(len(yy), len(xx))

	xspline = interp.RectBivariateSpline(xx, yy, Xo.T)
	yspline = interp.RectBivariateSpline(xx, yy, Yo.T)

	# Now, build the full pixel grid we want to interpolate...
	xx = np.arange(max(0, x0-margin), min(W, x1+margin+1))
	yy = np.arange(max(0, y0-margin), min(H, y1+margin+1))
	# And run the interpolator
	Xi = xspline(xx,yy).T
	Yi = yspline(xx,yy).T

	if len(Limages):
		m = L+1
		# Keep only in-bounds pixels
		I = np.flatnonzero((Xi >= m) * (Xi <= w-1-m) * (Yi >= m) * (Yi <= h-1-m))
		# Center of interpolation
		xi = Xi.flat[I]
		yi = Yi.flat[I]
		Xi = np.round(xi).astype(int)
		Yi = np.round(yi).astype(int)
		dx = Xi - xi
		dy = Yi - yi

		Xo,Yo = xx[I%(len(xx))], yy[I/(len(xx))]

		assert(np.all(Xi >= 0))
		assert(np.all(Yi >= 0))
		assert(np.all(Xo >= 0))
		assert(np.all(Yo >= 0))
		assert(np.all(Xi < w))
		assert(np.all(Yi < h))
		assert(np.all(Xo < W))
		assert(np.all(Yo < H))

		# Lanczos interpolation.
		# number of pixels
		nn = len(Yo)
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
					lacc   += fx * fy * im[Yi + oy, Xi + ox]
				fsum += fx*fy
		for lacc in laccs:
			lacc /= fsum

		rims = laccs

	else:
		rims = []

	return (Yo, Xo, Yi, Xi, rims)
