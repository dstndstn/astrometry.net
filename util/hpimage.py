import sys

from pylab import *
import pyfits
from numpy import *

from astrometry.util.healpix import *
from astrometry.util.sip import *
from astrometry.util.starutil_numpy import *
from astrometry.util.pyfits_utils import *

if __name__ == '__main__':
	args = sys.argv[1:]
	if len(args):
		base = args[0]
	else:
		base = 'img'
	imgfn = base + '.jpg'
	wcsfn = base + '.wcs'

	img = imread(imgfn)
	wcs = Sip(filename=wcsfn)

	if len(img.shape) == 3:
		IH,IW,NC = img.shape
	else:
		NC = 1
		IH,IW = img.shape
		img = img[:,:,newaxis]

	print 'img is', img.shape
	print 'wcs is', wcs

	# arcsec/pix
	pixscale = wcs.get_pixel_scale()
	res = 1.2
	nside = healpix_nside_for_side_length_arcmin(pixscale / 60.)
	nside = int(nside * res)
	print 'Nside', nside

	(ralo, rahi, declo, dechi) = wcs.radec_bounds()
	print 'radec bounds', ralo, rahi, declo, dechi
	ras = array([ralo, ralo, rahi, rahi])
	#(ralo+rahi)/2
	decs = array([declo, dechi, declo, dechi])
	#(declo+dechi)/2
	xyzs = radectoxyz(ras, decs)
	(N,nil) = xyzs.shape
	hpxs,hpys = [],[]
	for i in range(N):
		xyz = xyzs[i,:]
		x,y,z = xyz
		(bighp, hpx, hpy) = xyztohealpixf(x,y,z, 1)
		print 'radec', ras[i], decs[i]
		print 'big hp', bighp
		print 'hp x,y', hpx, hpy
		hpxs.append(hpx)
		hpys.append(hpy)

	hpxs,hpys = [],[]
	for x,y in [(0,0),(0,IH),(IW,IH),(IW,0)]:
		ra,dec = wcs.pixelxy2radec(x,y)
		(bighp, hpx, hpy) = radectohealpixf(ra, dec, 1)
		hpxs.append(hpx)
		hpys.append(hpy)
		print 'big hp:', bighp

	hpxlo,hpxhi = min(hpxs),max(hpxs)
	hpylo,hpyhi = min(hpys),max(hpys)

	print 'hp range', hpxlo, hpxhi, hpylo, hpyhi
	print 'hp pix:', nside*(hpxhi-hpxlo), nside*(hpyhi-hpylo)

	HW,HH = int(ceil(nside*(hpxhi-hpxlo))), int(ceil(nside*(hpyhi-hpylo)))

	xscale,yscale = (hpxhi-hpxlo)/float(HW), (hpyhi-hpylo)/float(HH)
	Himg = zeros((NC,HH,HW))
	hxs = hpxlo + arange(HW) * xscale
	hys = hpylo + arange(HH) * yscale
	print 'hxs', min(hxs), max(hxs)
	print 'hys', min(hys), max(hys)
	nans = 0
	oobs = 0
	for hi,hx in enumerate(hxs):
		print 'col', hi, 'of', len(hxs)
		for hj,hy in enumerate(hys):
			x,y,z = healpix_to_xyz(bighp, 1, hx, hy)
			px,py = wcs.xyz2pixelxy(x,y,z)
			if isnan(px) or isnan(py):
				nans += 1
				continue
			px = int(round(px))
			py = int(round(py))
			if px < 0 or px >= IW:
				oobs += 1
				continue
			if py < 0 or py >= IH:
				oobs += 1
				continue
			Himg[:, hj, hi] = img[py, px, :]
		#break
	print 'nans:', nans
	print 'oobs:', oobs
	p = pyfits.PrimaryHDU(Himg)
	pyfits_writeto(p, base + '-hp.fits')

