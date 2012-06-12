#import matplotlib
from matplotlib.patches import Circle, Ellipse
from pylab import gca, gcf, gci, axis, histogram2d, hist
from numpy import array, append, flatnonzero

import numpy as np
import pylab as plt

def setRadecAxes(ramin, ramax, decmin, decmax):
	rl,rh = ramin,ramax
	dl,dh = decmin,decmax
	rascale = np.cos(np.deg2rad((dl+dh)/2.))
	ax = [rh,rl, dl,dh]
	plt.axis(ax)
	plt.gca().set_aspect(1./rascale, adjustable='box', anchor='C')
	plt.xlabel('RA (deg)')
	plt.ylabel('Dec (deg)')
	return ax

import matplotlib.colors as mc
class ArcsinhNormalize(mc.Normalize):
 	def __init__(self, mean=None, std=None, **kwargs):
 		self.mean = mean
 		self.std = std
		mc.Normalize.__init__(self, **kwargs)

	def _map(self, X, out=None):
		Y = (X - self.mean) / self.std
		args = (Y,)
		if out is not None:
			args = args + (out,)
		return np.arcsinh(*args)

	def __call__(self, value, clip=None):
		# copied from Normalize since it's not easy to subclass
		if clip is None:
			clip = self.clip
		result, is_scalar = self.process_value(value)
		self.autoscale_None(result)
		vmin, vmax = self.vmin, self.vmax
		if vmin > vmax:
			raise ValueError("minvalue must be less than or equal to maxvalue")
		elif vmin == vmax:
			result.fill(0)	 # Or should it be all masked?	Or 0.5?
		else:
			vmin = float(vmin)
			vmax = float(vmax)
			if clip:
				mask = ma.getmask(result)
				result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
			# ma division is very slow; we can take a shortcut
			resdat = result.data
			self._map(resdat, resdat)
			vmin = self._map(vmin)
			vmax = self._map(vmax)
			resdat -= vmin
			resdat /= (vmax - vmin)
			result = np.ma.array(resdat, mask=result.mask, copy=False)
		if is_scalar:
			result = result[0]
		return result


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

# x, y0, y1
_redgreen_data =  {'red':   ((0.,  -100,  1),
							 #(0.5,  0,  0),
							 #(0.5,  0.1, 0),
							 (0.49, 0.1, 0),
							 (0.491, 0, 0),
							 (0.51,  0, 0),
							 (0.511,  0, 0.1),
							 (1.,   0, -100)),
				   'green': ((0.,  -100,  0),
							 #(0.5,  0,  0),
							 #(0.5,  0,  0.1),
							 (0.49, 0.1, 0),
							 (0.491, 0, 0),
							 (0.51,  0, 0),
							 (0.511,  0, 0.1),
							 (1.,   1, -100)),
				   'blue':  ((0.,  -100,  0),
							 (1.,   0, -100))}
redgreen = LinearSegmentedColormap('redgreen',   _redgreen_data)

def hist_ints(x, step=1, **kwargs):
	'''
	Creates a histogram of integers.  The number of bins is set to the
	range of the data (+1).  That is, each integer gets its own bin.
	'''
	kwargs['bins'] = x.max()/step - x.min()/step + 1
	kwargs['range'] = ( (x.min()/int(step))*step - 0.5,
						((x.max()/int(step))*step + 0.5) )
	return hist(x, **kwargs)

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

