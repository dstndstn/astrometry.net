# Copyright 2010 Dustin Lang and David W. Hogg, all rights reserved.

# Note: not thoroughly tested.

# style notes:
# ------------
# - column is x, row is y; this is at odds with some online documentation

from astrometry.util.pyfits_utils import *
from numpy import *
import pyfits
from glob import glob

def tsfield_get_node_incl(tsfield):
	hdr = tsfield[0].header
	node = deg2rad(hdr.get('NODE'))
	incl = deg2rad(hdr.get('INCL'))
	return (node, incl)

def radec_to_pixel(ra, dec, color, band, tsfield):
	mu, nu = radec_to_munu(ra, dec, tsfield)
	return munu_to_pixel(mu, nu, color, band, tsfield)

def pixel_to_radec(x, y, color, band, tsfield):
	mu, nu = pixel_to_munu(x, y, color, band, tsfield)
	return munu_to_radec(mu, nu, tsfield)

def munu_to_radec(mu, nu, tsfield):
	node, incl = tsfield_get_node_incl(tsfield)
	mu, nu = deg2rad(mu), deg2rad(nu)
	ra = node + arctan2(sin(mu - node) * cos(nu) * cos(incl) - sin(nu) * sin(incl), cos(mu - node) * cos(nu))
	dec = arcsin(sin(mu - node) * cos(nu) * sin(incl) + sin(nu) * cos(incl))
	ra, dec = rad2deg(ra), rad2deg(dec)
	ra += (360. * (ra < 0))
	return (ra, dec)

def radec_to_munu(ra, dec, tsfield):
	node, incl = tsfield_get_node_incl(tsfield)
	ra, dec = deg2rad(ra), deg2rad(dec)
	mu = node + arctan2(sin(ra - node) * cos(dec) * cos(incl) + sin(dec) * sin(incl), cos(ra - node) * cos(dec))
	nu = arcsin(-sin(ra - node) * cos(dec) * sin(incl) + sin(dec) * cos(incl))
	mu, nu = rad2deg(mu), rad2deg(nu)
	mu += (360. * (mu < 0))
	return (mu, nu)

def prime_to_pixel(xprime, yprime,  color, band, tsfield):
	T = fits_table(tsfield[1].data)
	T = T[0]
	color0 = T.ricut[band]
	g0, g1, g2, g3 = T.drow0[band], T.drow1[band], T.drow2[band], T.drow3[band]
	h0, h1, h2, h3 = T.dcol0[band], T.dcol1[band], T.dcol2[band], T.dcol3[band]
	px, py = T.csrow[band], T.cscol[band]
	qx, qy = T.ccrow[band], T.cccol[band]

	# #$(%*&^(%$%*& bad documentation.
	(px,py) = (py,px)
	(qx,qy) = (qy,qx)

	color  = atleast_1d(color)
	color0 = atleast_1d(color0)
	# HACK HACK HACK FIXME
	qx = qx * ones_like(xprime)
	qy = qy * ones_like(yprime)
	print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
	xprime -= where(color < color0, px * color, qx)
	yprime -= where(color < color0, py * color, qy)

	# Now invert:
	#   yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
	#   xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
	x = xprime - h0
	# dumb-ass Newton's method
	dx = 1.
	while max(abs(atleast_1d(dx))) > 1e-10:
		xp    = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
		dxpdx = 1 +      h1     + h2 * 2*x +  h3 * 3*x**2
		dx = (xprime - xp) / dxpdx
		print 'Max Newton dx', max(abs(dx))
		x += dx
	y = yprime - (g0 + g1 * x + g2 * x**2 + g3 * x**3)
	return (x, y)

def pixel_to_prime(x, y, color, band, tsfield):
	T = fits_table(tsfield[1].data)
	T = T[0]

	# Secret decoder ring:
	#  http://www.sdss.org/dr7/products/general/astrometry.html
	# (color)0 is called riCut;
	# g0, g1, g2, and g3 are called
	#    dRow0, dRow1, dRow2, and dRow3, respectively;
	# h0, h1, h2, and h3 are called
	#    dCol0, dCol1, dCol2, and dCol3, respectively;
	# px and py are called csRow and csCol, respectively;
	# and qx and qy are called ccRow and ccCol, respectively.
	color0 = T.ricut[band]
	g0, g1, g2, g3 = T.drow0[band], T.drow1[band], T.drow2[band], T.drow3[band]
	h0, h1, h2, h3 = T.dcol0[band], T.dcol1[band], T.dcol2[band], T.dcol3[band]
	px, py = T.csrow[band], T.cscol[band]
	qx, qy = T.ccrow[band], T.cccol[band]
	# #$(%*&^(%$%*& bad documentation.
	(px,py) = (py,px)
	(qx,qy) = (qy,qx)

	yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
	xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3

	#if color < color0:
	#	xprime += px * color
	#	yprime += py * color
	#else:
	#	xprime += qx
	#	yprime += qy
	color  = atleast_1d(color)
	color0 = atleast_1d(color0)
	qx = qx * ones_like(x)
	qy = qy * ones_like(y)
	print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
	xprime += where(color < color0, px * color, qx)
	yprime += where(color < color0, py * color, qy)
	return (xprime, yprime)

def pixel_to_munu(x, y, color, band, tsfield):
	T = fits_table(tsfield[1].data)
	T = T[0]
	(xprime, yprime) = pixel_to_prime(x, y, color, band, tsfield)
	a, b, c = T.a[band], T.b[band], T.c[band]
	d, e, f = T.d[band], T.e[band], T.f[band]
	mu = a + b * yprime + c * xprime
	nu = d + e * yprime + f * xprime
	return (mu, nu)

def munu_to_pixel(mu, nu, color, band, tsfield):
	T = fits_table(tsfield[1].data)
	T = T[0]
	a, b, c = T.a[band], T.b[band], T.c[band]
	d, e, f = T.d[band], T.e[band], T.f[band]
	determinant = b * f - c * e
	B = f / determinant
	C = -c / determinant
	E = -e / determinant
	F = b / determinant
	yprime = B * (mu - a) + C * (nu - d)
	xprime = E * (mu - a) + F * (nu - d)
	return prime_to_pixel(xprime, yprime, color, band, tsfield)

if __name__ == '__main__':
	tsfield = pyfits.open(glob('tsField-002830-6-*-0398.fit')[0])

	x,y = 0,0
	band = array([2,3])
	color = 0.
	rd = pixel_to_radec(x, y, color, band, tsfield)
	print rd

	N = 10
	x = numpy.random.uniform(2048, size=(N,))
	y = numpy.random.uniform(1489, size=(N,))
	color = numpy.random.uniform(-1, 4, size=(N,))
	band = 2
	rd = pixel_to_radec(x, y, color, band, tsfield)
	print rd

	m,n = numpy.random.uniform(360, size=N), numpy.random.uniform(-90, 90, size=N)
	r,d = munu_to_radec(m, n, tsfield)
	mu,nu = radec_to_munu(r, d, tsfield)
	print mu-m, nu-n

	x,y = numpy.random.uniform(2048, size=N), numpy.random.uniform(1489, 90, size=N)
	mu,nu = pixel_to_munu(x, y, color, band, tsfield)
	xx,yy = munu_to_pixel(mu, nu, color, band, tsfield)
	print 'diff:', xx-x, yy-y


	x,y = numpy.random.uniform(2048, size=N), numpy.random.uniform(1489, 90, size=N)
	ra,dec = pixel_to_radec(x, y, color, band, tsfield)
	xx,yy = radec_to_pixel(ra, dec, color, band, tsfield)
	rr,dd = pixel_to_radec(xx, yy, color, band, tsfield)
	print 'diff in pix:', xx-x, yy-y
	print 'diff in ra,dec:', rr-ra, dd-dec


	from astrometry.util.sip import *
	wcs = Sip('fpC-002830-r6-0398.fit')

	N = 800
	O = fits_table('tsObj-002830-6-0-0398.fit')
	ra,dec = O.ra, O.dec
	print 'ra,dec', ra.shape, dec.shape
	band = 2
	ra = ra[:N]
	dec = dec[:N]
	X,Y = O.colc[:,band], O.rowc[:,band]
	X = X[:N]
	Y = Y[:N]
	print 'x', X.shape
	rmag = O.psfcounts[:,2]
	imag = O.psfcounts[:,3]
	color = (rmag - imag)[:N]
	#color = zeros_like(X)
	xx,yy = radec_to_pixel(ra, dec, color, band, tsfield)
	print 'xx', xx.shape
	rr,dd = pixel_to_radec(X, Y, color, band, tsfield)
	print 'dxy', xx-X, yy-Y
	print 'dradec', ra-rr, dec-dd


	# with color:
	# 14 mpix X, 14 mpix Y

	# color=0:
	#  2 mpix X, 14 mpix Y

	print 'MAD:', median(abs(xx-X)), median(abs(yy-Y))

	if False:
		for Xi,Yi,xi,yi,ri,di in zip(X,Y,xx,yy,ra,dec):
			wxi,wyi = wcs.radec2pixelxy(ri,di)
			print ('  X,Y (%8.2f,%8.2f) xx,yy (%8.2f, %8.2f), dx,dy (%8.2f, %8.2f), wcs xy (%8.2f, %8.2f) wcs dxy (%8.2f, %8.2f)' %
				   (Xi, Yi, xi, yi, (Xi-xi), (Yi-yi), wxi, wyi, Xi-wxi, Yi-wyi))

	from pylab import *

	clf()
	plot(X, Y, 'k.')
	plot(xx, yy, 'rx')
	plot(vstack((X,xx)), array((Y,yy)), '-', color='0.5')
	savefig('dxy.png')
	
	#rd = array([wcs.pixelxy2radec(x,y) for x,y in zip(X,Y)])
	#r1 = rd[:,0]
	#d1 = rd[:,1]
	#print 'delta radec', r1

	if False:
		for band in range(5):
			X,Y = O.colc[:,band], O.rowc[:,band]
			color = zeros_like(X)
			xx,yy = radec_to_pixel(ra, dec, color, band, tsfield)
			print 'xx', xx.shape
			rr,dd = pixel_to_radec(X, Y, color, band, tsfield)
			print 'dxy', xx-X, yy-Y
			print 'dradec', ra-rr, dec-dd
