import sys
import pyfits

from math import *
from numpy import *
from pylab import *
from scipy.ndimage.filters import *

def normalized_hough(x, y, imgw, imgh, rlo, rhi, tlo, thi, nr, nt):
	houghimg = zeros((nr, nt)).astype(int)

	tstep = (thi - tlo) / float(nt)
	rstep = (rhi - rlo) / float(nr)
	tt = tlo + (arange(nt) + 0.5) * tstep
	cost = cos(tt)
	sint = sin(tt)

	# For each point, accumulate into the Hough transform image...
	for (xi,yi) in zip(x, y):
		rr = xi * cost + yi * sint
		ri = floor((rr - rlo) / rstep).astype(int)
		I = (ri >= 0) * (ri < nr)
		houghimg[ri[I], I] += 1

	houghnorm = zeros((nr, nt)).astype(float)
	rr = rlo + (arange(nr) + 0.5) * rstep
	for ti in range(nt):
		t = tlo + (ti + 0.5) * tstep
		#print 'ti=', ti, 't=', t
		(x0,x1,y0,y1) = clip_to_image(rr, t, imgw, imgh)
		dist = sqrt((x0 - x1)**2 + (y0 - y1)**2)
		houghnorm[:, ti] = dist

	# expected number of points... rstep is the width of the slice; len(x)/A is the source density.
	houghnorm *= (rstep * len(x) / (imgw*imgh))

	return (houghimg, houghnorm, rr, tt, rstep, tstep)


def clip_to_image(r, t, imgw, imgh):
	eps = 1e-9
	if abs(t) < eps or abs(t-pi) < eps:
		# near-vertical.
		s = (abs(t) < eps) and 1 or -1
		y0 = 0
		y1 = ((r*s >= 0) * (r*s < imgw)) * imgh
		x0 = x1 = clip(r, 0, imgw)
		return (x0, x1, y0, y1)
	m = -cos(t)/sin(t)
	b = r/sin(t)
	#print 'clip:'
	#print 'r=', r
	#print 't=', t
	#print 'm=', m
	#print 'b=', b
	x0 = 0
	x1 = imgw
	y0 = clip(b + m*x0, 0, imgh)
	y1 = clip(b + m*x1, 0, imgh)
	x0 = clip((y0 - b) / m, 0, imgw)
	x1 = clip((y1 - b) / m, 0, imgw)
	y0 = clip(b + m*x0, 0, imgh)
	y1 = clip(b + m*x1, 0, imgh)
	#print 'got:', (x0, x1, y0, y1)
	return (x0, x1, y0, y1)

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

	(houghimg, houghnorm, rr, tt, rstep, tstep) = normalized_hough(x, y, imgw, imgh, Rmin, Rmax, 0, pi, nr, nt)

	clf()
	imshow(houghimg, **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('hough.png')

	clf()
	imshow(houghnorm, **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('norm.png')

	hnorm = houghimg / maximum(houghnorm, 1)
	clf()
	imshow(hnorm, **imshowargs)
	xlabel('Theta')
	ylabel('Radius')
	colorbar()
	savefig('hnorm.png')

	thresh1 = 2.
	thresh2 = 10.

	I = find(hnorm.ravel() >= thresh1)
	print '%i peaks are above the coarse threshold' % len(I)
	bestri = I / nt
	bestti = I % nt

	a=axis()
	for (ri,ti) in zip(bestri,bestti):
		plot([ti-2, ti-2, ti+2, ti+2, ti-2], [ri-2, ri+2, ri+2, ri-2, ri-2], 'r-')
	axis(a)
	savefig('zooms.png')

	clf()
	plot(x,y,'r.')
	for (ri,ti) in zip(bestri,bestti):
		r = rr[ri]
		t = tt[ti]
		(x0,x1,y0,y1) = clip_to_image(r, t, imgw, imgh)
		plot([x0,x1],[y0,y1], 'b-')
	savefig('xy2.png')

	boxsize = 2
	nr2 = (boxsize * 2) * 5 + 2
	nt2 = nr2

	#clf()
	bestrt = []
	xys = []
	for (ri,ti) in zip(bestri,bestti):
		#print 'ri=',ri, 'ti=', ti
		r = rr[ri]
		t = tt[ti]
		#print 'r=', r, 't=', t
		#print 'testing r range', rr[max(ri-boxsize, 0)], 'to', rr[min(ri+boxsize,nr-1)]
		#print 'testing t range', tt[max(ti-boxsize, 0)], 'to', tt[min(ti+boxsize,nt-1)]
		(subh, subhnorm, subrr, subtt, subrstep,
		 subtstep) = normalized_hough(x, y, imgw, imgh,
									  rr[max(ri-boxsize, 0)], rr[min(ri+boxsize,nr-1)],
									  tt[max(ti-boxsize, 0)], tt[min(ti+boxsize,nt-1)],
									  nr2, nt2)
		#print 'tried r values', subrr
		#print 'tried t values', subtt
		#print 'total r range', rr[min(ri+boxsize,nr-1)] - rr[max(ri-boxsize, 0)]
		#print 'tried r range', subrr.max() - subrr.min()
		#print 'total t range', tt[min(ti+boxsize,nt-1)] - tt[max(ti-boxsize, 0)]
		#print 'tried t range', subtt.max() - subtt.min()

		subnormed = subh / maximum(subhnorm,1)

		#subplot(3,3,i+1)
		#imshow(subnormed, vmin=0, **imshowargs)

		I = find((subnormed).ravel() >= thresh2)
		for i in I:
			bestsubri = i / nt2
			bestsubti = i % nt2
			X = clip_to_image(subrr[bestsubri], subtt[bestsubti], imgw, imgh)
			xys.append(X)
			bestrt.append((subrr[bestsubri], subtt[bestsubti]))
	#savefig('subhough.png')

	# truncate to remove duplicates...
	bestrt = [(1e-3 * int(r*1e3), 1e-5 * int(t*1e5)) for (r,t) in bestrt]
	bestrt = list(set(bestrt))
	bestrt.sort()
	print 'In finer Hough grid: bests are', bestrt

	print 'Found %i peaks' % len(bestrt)

	clf()
	subplot(1,1,1)
	plot(x,y,'r.')
	for (x0,x1,y0,y1) in xys:
		plot([x0,x1],[y0,y1],'b-')
	savefig('xy3.png')

	clf()
	plot(x,y,'r.')
	tt = array([t for (r,t) in bestrt])
	rr = array([r for (r,t) in bestrt])
	cost = cos(tt)
	sint = sin(tt)
	keep = []
	for i,(xi,yi) in enumerate(zip(x, y)):
		thisr = xi * cost + yi * sint
		keep.append(not any(abs(thisr - rr) < rstep/2.))
	keep = array(keep)

	plot(x[keep == False], y[keep == False], 'b.')
	savefig('xy4.png')


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

