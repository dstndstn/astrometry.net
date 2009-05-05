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

	#nt = 360
	#nr = 360
	nt = 180
	nr = 180

	Rmax = sqrt(imgw**2 + imgh**2)
	Rmin = -Rmax
	tstep = pi / float(nt)
	rstep = (Rmax-Rmin) / float(nr)

	ti = arange(nt)
	#thetas = linspace(0, pi, num=nt, endpoint=False)
	thetas = (ti+0.5) * tstep
	cost = cos(thetas)
	sint = sin(thetas)




	houghimg = zeros((nr, nt)).astype(float)

	for (xi,yi) in zip(x, y):
		rr = xi * cost + yi * sint
		ri = floor((rr - Rmin) / rstep).astype(int)
		houghimg[ri, ti] += 1.

	#houghimg /= max(1, houghimg.max())

	clf()
	#hfilt = gaussian_filter(houghimg, 1.0)
	imshow(houghimg, **imshowargs)
	#imshow(hfilt, **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('hough.png')

	clf()
	tlo = thetas[45]
	thi = thetas[46]
	tnext = thetas[47]
	rlo = Rmin + rstep * 100
	rhi = Rmin + rstep * 101
	rnext = Rmin + rstep * 102
	for t in [tlo, thi]:
		for r in [rlo, rhi]:
			xs = array([0, imgw])
			ys = -tan(t) * xs + r/sin(t)
			plot(xs, ys, 'r--')
	for t in [thi, tnext]:
		for r in [rhi, rnext]:
			xs = array([0, imgw])
			ys = -tan(t) * xs + r/sin(t)
			plot(xs, ys, 'b:')
	savefig('rt.png')

	# computing the exact image area covered by each hough bin is hard,
	# but approximating it by the center is easier.
	houghnorm2 = zeros((nr, nt)).astype(float)
	r = Rmin + (arange(0, nr) + 0.5) * rstep
	#r = array([Rmin + (nr-1 + 0.5) * rstep])

	clf()
	for ti in range(nt):
		print ti
		t = (ti+0.5) * tstep
		m = -cos(t)/sin(t)
		b = r/sin(t)

		#print 'ti=',ti, 't=', t, 'm=', m, 'b=', b

		#if m > 0:
		x0 = zeros(*r.shape)
		x1 = imgw * ones(*r.shape)
		#else:
		#	x0 = imgw * ones(*r.shape)
		#	x1 = zeros(*r.shape)

		y0 = clip(b + m*x0, 0, imgh)
		y1 = clip(b + m*x1, 0, imgh)

		x0 = clip((y0 - b) / m, 0, imgw)
		x1 = clip((y1 - b) / m, 0, imgw)

		y0 = clip(b + m*x0, 0, imgh)
		y1 = clip(b + m*x1, 0, imgh)

		#print 'x0=', x0, 'x1=', x1
		#print 'y0=', y0, 'y1=', y1

		#plot([x0,x1],[y0,y1], 'r-')
		#axis([-100, imgw+100, -100, imgh+100])
		#savefig('norm2-%03i.png' % ti)

		dist = sqrt((x0 - x1)**2 + (y0 - y1)**2)
		houghnorm2[:, ti] = dist;


	houghnorm2 *= (rstep / (imgw*imgh) * len(x))
	#axis([-100, imgw+100, -100, imgh+100])
	imshow(houghnorm2, **imshowargs)
	colorbar()
	savefig('norm2.png')
	print 'norm2'


	clf()
	imshow(houghimg / maximum(houghnorm2, 1), **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('hnorm.png')


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

