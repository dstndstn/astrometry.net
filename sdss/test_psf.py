import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import sys
import os
from astrometry.sdss.dr9 import *
from astrometry.sdss import *
from astrometry.util.plotutils import *

if __name__ == '__main__':
	run, camcol, field = 1752, 3, 163
	band ='r'
	bandnum = band_index(band)

	datadir = os.path.join(os.path.dirname(__file__), 'testdata')

	sdss = DR9(basedir=datadir)
	psfield = sdss.readPsField(run, camcol, field)

	ps = PlotSequence('klpsf')

	# These psf*.fits files were produced on the NYU system via:
	#
	#  psfield = mrdfits('psField-001752-3-0163.fit', 3)
	#  psfimage = sdss_psf_recon(psfield, 1000., 0.)
	#  mwrfits,psfimage,'psf1k0.fits',/CREATE
	#
	for x,y,fn in [(0.,0.,'psf00.fits'),
				   (0., 1000., 'psf01k.fits'),
				   (1000., 0., 'psf1k0.fits'),
				   (0., 2000., 'psf02k.fits'),
				   (2000., 0., 'psf2k0.fits'),
				   (600.,500.,'psf.fits')]:

		psf0 = pyfits.open(os.path.join(datadir, fn))[0].data
		psf = psfield.getPsfAtPoints(bandnum, x, y)
		psf = psf.astype(np.float32)

		def show(im):
			plt.imshow(np.log10(np.maximum(im, 1e-4)), interpolation='nearest', origin='lower')
		
		plt.clf()
		show(psf0)
		plt.gray()
		plt.colorbar()
		plt.title('IDL: %.1f, %.1f' % (x,y))
		ps.savefig()

		plt.clf()
		show(psf)
		plt.colorbar()
		plt.title('Me: %.1f, %.1f' % (x,y))
		ps.savefig()

		plt.clf()
		plt.imshow(psf - psf0, interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.title('Diff: %.1f, %.1f' % (x,y))
		ps.savefig()

		diff = psf - psf0

		print 'Diff:', diff.min(), diff.max()
		rms = np.sqrt(np.mean(diff**2))
		print 'RMS:', rms
		assert(np.all(np.abs(diff) < 2e-5))
		assert(rms < 1e-6)
