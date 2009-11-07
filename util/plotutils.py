#import matplotlib
from matplotlib.patches import Circle, Ellipse
from pylab import gca, gcf, gci, axis, histogram2d
from numpy import array, append, flatnonzero

from matplotlib.colors import LinearSegmentedColormap

# a colormap that goes from white to black: the opposite of matplotlib.gray()
antigray = LinearSegmentedColormap('antigray',
								   {'red':   ((0., 1, 1), (1., 0, 0)),
									'green': ((0., 1, 1), (1., 0, 0)),
									'blue':  ((0., 1, 1), (1., 0, 0))})

bluegrayred = LinearSegmentedColormap('bluegrayred',
									  {'red':   ((0., -1, 0),
												 (1., 1, -1)),
									   'green': ((0., -1,   0),
												 (0.5,0.5, 0.5),
												 (1., 0, -1)),
									   'blue':  ((0., -1, 1),
												 (1., 0, -1))})

def hist_ints(x, *args, **kwargs):
	'''
	Creates a histogram of integers.  The number of bins is set to the
	range of the data (+1).  That is, each integer gets its own bin.
	'''
	kwargs['bins'] = x.max() - x.min() + 1
	kwargs['range'] = (x.min() - 0.5, x.max() + 0.5)
	return hist(x, *args, **kwargs)

def hist2d_with_outliers(x, y, xbins, ybins, nout):
	'''
	Creates a 2D histogram from the given data, and returns a list of
	the indices in the data of points that lie in low-occupancy cells
	(where the histogram counts is < "nout").

	The "xbins" and "ybins" arguments are passed to numpy.histogram2d.

	You probably want to show the histogram with:

	  (H, outliers, xe, ye) = hist2d_with_outliers(x, y, 10, 10, 10)
	  imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto')
	  plot(x[outliers], y[outliers], 'r.')

	Returns: (H, outliers, xe, ye)

	  H: 2D histogram image
	  outliers: array of integer indices of the outliers
	  xe: x edges chosen by histgram2d
	  ye: y edges chosen by histgram2d
	  
	'''
	# returns (density image, indices of outliers)
	(H,xe,ye) = histogram2d(x, y, (xbins,ybins))
	Out = array([]).astype(int)
	for i in range(len(xe)-1):
		for j in range(len(ye)-1):
			if H[i,j] > nout:
				continue
			if H[i,j] == 0:
				continue
			H[i,j] = 0
			Out = append(Out, flatnonzero((x >= xe[i]) *
										  (x <  xe[i+1]) *
										  (y >= ye[j]) *
										  (y <  ye[j+1])))
	return (H.T, Out, xe, ye)


# You probably want to set the keyword radius=R
def circle(xy=None, x=None, y=None, **kwargs):
	if xy is None:
		if x is None or y is None:
			raise 'circle: need x and y'
		xy = array([x,y])
	c = Circle(xy=xy, **kwargs)
	a=gca()
	c.set_clip_box(a.bbox)
	a.add_artist(c)
	return c
	
def ellipse(xy=None, x=None, y=None, **kwargs):
	if xy is None:
		if x is None or y is None:
			raise 'ellipse: need x and y'
		xy = array([x,y])
	c = Ellipse(xy=xy, **kwargs)
	a=gca()
	c.set_clip_box(a.bbox)
	a.add_artist(c)
	return c

# return (pixel width, pixel height) of the axes area.
def get_axes_pixel_size():
	dpi = gcf().get_dpi()
	figsize = gcf().get_size_inches()
	axpos = gca().get_position()
	pixw = figsize[0] * dpi * axpos.width
	pixh = figsize[1] * dpi * axpos.height
	return (pixw, pixh)

    # test:
	if False:
		figure(dpi=100)
		(w,h) = get_axes_pixel_size()
		# not clear why this is required...
		w += 1
		h += 1
		img = zeros((h,w))
		img[:,::2] = 1.
		img[::2,:] = 1.
		imshow(img, extent=(0,w,0,h), aspect='auto', cmap=antigray)
		xlim(0,w)
		ylim(0,h)
		savefig('imtest.png')
		sys.exit(0)

# returns (x data units per pixel, y data units per pixel)
# given the current plot range, figure size, and axes position.
def get_pixel_scales():
	a = axis()
	(pixw, pixh) = get_axes_pixel_size()
	return ((a[1]-a[0])/float(pixw), (a[3]-a[2])/float(pixh))

def set_image_color_percentiles(image, plo, phi):
	# hackery...
	I = image.copy().ravel()
	I.sort()
	N = len(I)
	mn = I[max(0, int(round(plo * N / 100.)))]
	mx = I[min(N-1, int(round(phi * N / 100.)))]
	gci().set_clim(mn, mx)
	return (mn,mx)

