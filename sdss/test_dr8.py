import matplotlib
matplotlib.use('Agg')
import pylab as plt

import sys
from astrometry.sdss.dr8 import *

def test_astrans(sdss):
	r,c,f,b = 4623, 1, 203, 'r'
	bandnum = band_index(b)
	sdss.retrieve('frame', r, c, f, b)
	frame = sdss.readFrame(r, c, f, b)
	astrans = frame.getAsTrans()
	sdss.retrieve('photoObj', r, c, f)
	obj = sdss.readPhotoObj(r, c, f)
	tab = obj.getTable()
	#tab.about()
	x,y = tab.colc[:,bandnum], tab.rowc[:,bandnum]
	ra,dec = tab.ra, tab.dec

	r2,d2 = astrans.pixel_to_radec(x, y)
	plt.clf()
	plt.plot(ra, dec, 'r.')
	plt.plot(r2, d2, 'bo', mec='b', mfc='none')
	plt.savefig('rd.png')

	x2,y2 = astrans.radec_to_pixel(ra, dec)
	plt.clf()
	plt.plot(x, y, 'r.')
	plt.plot(x2, y2, 'bo', mec='b', mfc='none')
	plt.savefig('xy.png')
	

if __name__ == '__main__':
	sdss = DR8()
	test_astrans(sdss)
	sys.exit(0)

	fnew  = sdss.readFrame(4623, 1, 203, 'r', filename='frame-r-004623-1-0203.fits')
	print 'fnew:', fnew
	forig = sdss.readFrame(4623, 1, 203, 'r', 'frame-r-004623-1-0203.fits.orig')
	print 'forig:', forig

	frame = sdss.readFrame(3712, 3, 187, 'r')
	print 'frame:', frame
	img = frame.getImage()
	print '  image', img.shape

	fpobj = sdss.readFpObjc(6581, 2, 135)
	print 'fpobj:', fpobj

	fpm = sdss.readFpM(6581, 2, 135, 'i')
	print 'fpm:', fpm

	psf = sdss.readPsField(6581, 2, 135)
	print 'psfield:', psf


	
