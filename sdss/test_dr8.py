
from astrometry.sdss.dr8 import *

if __name__ == '__main__':
	sdss = DR8()

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

