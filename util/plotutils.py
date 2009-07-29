#import matplotlib
from matplotlib.patches import Circle, Ellipse
from pylab import gca,gcf,gci
from numpy import array

from matplotlib.colors import LinearSegmentedColormap

# a colormap that goes from white to black: the opposite of matplotlib.gray()
antigray = LinearSegmentedColormap('antigray',
								   {'red':   ((0., 1, 1), (1., 0, 0)),
									'green': ((0., 1, 1), (1., 0, 0)),
									'blue':  ((0., 1, 1), (1., 0, 0))})

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


def set_image_color_percentiles(image, plo, phi):
	# hackery...
	I = image.copy().ravel()
	I.sort()
	N = len(I)
	mn = I[max(0, int(round(plo * N / 100.)))]
	mx = I[min(N-1, int(round(phi * N / 100.)))]
	gci().set_clim(mn, mx)


