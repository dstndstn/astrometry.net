from astrometry.util.pyfits_utils import *
from astrometry.util.sdss_psf import *
from pylab import *

if __name__ == '__main__':
	
	psfield = pyfits.open('psField-000745-2-0518.fit')
	#psField-002830-6-0398.fit

	X = linspace(0, 2048, 7)
	Y = linspace(0, 1489, 5)

	band=0
	x,y = 0,0
	psf = sdss_psf_at_points(psfield[band+1], x, y)
	psfshape = psf.shape
	h,w = psfshape
	# Core:
	#s = 4
	#vmn,vmx = 0,0.12
	s = 7
	vmn,vmx = 0,0.012

	psfcut = slice(h/2-s, h/2+s+1),slice(w/2-s, w/2+s+1)
	
	for band in range(5):
		clf()
		ploti = 1

		for iy,y in enumerate(Y):
			for ix,x in enumerate(X):
				psf = sdss_psf_at_points(psfield[band+1], x, y)
				psfshape = psf.shape
				subplot(len(Y), len(X), ploti)
				ploti += 1
				print 'psf range:', psf.min(), psf.max(), psf.sum()
				imshow(psf[psfcut], interpolation='nearest', origin='lower', vmin=vmn, vmax=vmx)
				xticks([],[])
				yticks([],[])
		subplots_adjust(wspace=0.05, hspace=0.05)
		savefig('psf-%i-kl.png' % band)
		
		dgp = sdss_dg_psf_params(psfield, band)
		#print 'a,s1,b,s2', a,s1,b,s2
		clf()
		ploti = 1
		for iy,y in enumerate(Y):
			for ix,x in enumerate(X):
				psf = sdss_dg_psf(dgp, psfshape)
				subplot(len(Y), len(X), ploti)
				ploti += 1
				print 'psf range:', psf.min(), psf.max(), psf.sum()
				imshow(psf[psfcut], interpolation='nearest', origin='lower', vmin=vmn, vmax=vmx)
				xticks([],[])
				yticks([],[])

		subplots_adjust(wspace=0.05, hspace=0.05)
		savefig('psf-%i-dg.png' % band)
		
