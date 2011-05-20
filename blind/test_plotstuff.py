import matplotlib
if __name__ == '__main__':
	matplotlib.use('Agg')
import unittest

import pyfits
import numpy as np
import pylab as plt

from astrometry.blind.plotstuff import *

class TestPlotstuff(unittest.TestCase):

	def setUp(self):
		np.random.seed(42)

	
	def test_image_wcs(self):
		'''
		Round-trip an image through a WCS.
		'''
		W,H = 100,100
		I = np.random.uniform(size=(H,W))
		imfn = 'test-plotstuff-1.fits'
		pyfits.writeto(imfn, I, clobber=True)

		wcs = anwcs_create_box(43., 11., 1., W, H)
		anwcs_rotate_wcs(wcs, 32.)
		wcsfn = 'test-plotstuff-1.wcs'
		anwcs_write(wcs, wcsfn)

		plot = Plotstuff()
		plot.outformat = PLOTSTUFF_FORMAT_PNG
		plot.size = (W, H)
		plot.wcs_file = wcsfn
		plot.color = 'black'
		plot.plot('fill')

		im = plot.image
		plot_image_set_wcs(im, wcsfn, 0)
		im.image_high = 1.
		im.image_low = 0.
		plot_image_set_filename(im, imfn)
		plot.plot('image')
		plotfn = 'test-plotstuff-1.png'
		plot.write(plotfn)

		I2 = plt.imread(plotfn)
		print I2.shape
		I2 = I2[:,:,0]

		plt.clf()
		plt.subplot(2,2,1)
		plt.imshow(I, vmin=0, vmax=1)
		plt.subplot(2,2,2)
		plt.imshow(I2, vmin=0, vmax=1)
		plt.subplot(2,2,3)
		plt.imshow(I2 - I, vmin=-1, vmax=1)
		plt.savefig('test-plotstuff-1b.png')

		md = np.max(np.abs((I2-I).ravel()))
		print 'max diff', md
		self.assertTrue(md < 1./255.)

		
if __name__ == '__main__':
	unittest.main()
	   
