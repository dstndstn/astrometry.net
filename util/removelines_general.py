import sys
import pyfits

from math import *
from numpy import *
from pylab import *
from scipy.ndimage.filters import *

def removelines_general(infile, outfile, **kwargs):
	p = pyfits.open(infile)
	xy = p[1].data
	hdr = p[1].header
	x = xy.field('X')
	y = xy.field('Y')

	imshowargs = { 'interpolation':'nearest', 'origin':'lower' }

	imgw = int(ceil(max(x) - min(x)))
	imgh = int(ceil(max(y) - min(y)))

	x -= min(x)
	y -= min(y)

	clf()
	plot(x,y,'r.')
	savefig('xy.png')

	nt = 360
	nr = 360
	#nt = 180
	#nr = 180

	Rmax = sqrt(imgw**2 + imgh**2)
	Rmin = -Rmax
	tstep = pi / float(nt)
	rstep = (Rmax-Rmin) / float(nr)

	ti = arange(nt)
	thetas = (ti+0.5) * tstep
	cost = cos(thetas)
	sint = sin(thetas)

	houghimg = zeros((nr, nt)).astype(float)

	# For each point, accumulate into the Hough transform image...
	for (xi,yi) in zip(x, y):
		rr = xi * cost + yi * sint
		ri = floor((rr - Rmin) / rstep).astype(int)
		houghimg[ri, ti] += 1.

	clf()
	imshow(houghimg, **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('hough.png')

	# computing the exact image area covered by each hough bin is hard,
	# but approximating it by the center is easier.
	houghnorm2 = zeros((nr, nt)).astype(float)
	r = Rmin + (arange(0, nr) + 0.5) * rstep
	for ti in range(nt):
		t = (ti+0.5) * tstep
		m = -cos(t)/sin(t)
		b = r/sin(t)
		x0 = zeros(*r.shape)
		x1 = imgw * ones(*r.shape)
		y0 = clip(b + m*x0, 0, imgh)
		y1 = clip(b + m*x1, 0, imgh)
		x0 = clip((y0 - b) / m, 0, imgw)
		x1 = clip((y1 - b) / m, 0, imgw)
		y0 = clip(b + m*x0, 0, imgh)
		y1 = clip(b + m*x1, 0, imgh)
		dist = sqrt((x0 - x1)**2 + (y0 - y1)**2)
		houghnorm2[:, ti] = dist;

	# expected number of points... rstep is the width of the slice; len(x)/A is the source density.
	houghnorm2 *= (rstep / (imgw*imgh) * len(x))
	clf()
	imshow(houghnorm2, **imshowargs)
	colorbar()
	savefig('norm.png')

	hnorm = houghimg / maximum(houghnorm2, 1)
	clf()
	imshow(hnorm, **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('hnorm.png')

	clf()
	plot(x,y,'r.')
	k = 10
	I = argsort(hnorm.ravel())[-k:]
	ri = I / nt
	ti = I % nt
	print ri, ti
	rr = Rmin + (ri + 0.5) * rstep
	tt = (ti+0.5) * tstep

	for (r,t) in zip(rr,tt):
		m = -cos(t)/sin(t)
		b = r/sin(t)
		x0 = 0
		x1 = imgw
		y0 = clip(b + m*x0, 0, imgh)
		y1 = clip(b + m*x1, 0, imgh)
		x0 = clip((y0 - b) / m, 0, imgw)
		x1 = clip((y1 - b) / m, 0, imgw)
		y0 = clip(b + m*x0, 0, imgh)
		y1 = clip(b + m*x1, 0, imgh)
		plot([x0,x1],[y0,y1], 'b-')

	savefig('xy2.png')


def exact_hough_normalization():
	houghnorm = zeros((nr, nt)).astype(float)
	[xx,yy] = meshgrid(arange(imgw), arange(imgh))
	yyflat = yy.ravel()
	xxflat = xx.ravel()
	for ti in range(nt):
		print ti
		t = (ti+0.5) * tstep
		rr = xxflat * cos(t) + yyflat * sin(t)
		ri = floor((rr - Rmin) / rstep).astype(int)
		(counts, nil) = histogram(ri, range(0, nr+1))
		houghnorm[:, ti] += counts
	clf()
	imshow(houghnorm, **imshowargs)
	colorbar()
	savefig('houghnorm.png')
	


if __name__ == '__main__':
	if (len(sys.argv) == 3):
		infile = sys.argv[1]
		outfile = sys.argv[2]
		rtncode = removelines_general(infile, outfile)
		sys.exit(rtncode)
	else:
		print 'Usage: %s <input-file> <output-file>' % sys.args[0]

