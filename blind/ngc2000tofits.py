import pyfits
from numpy import *
from ngc2000 import ngc2000, ngc2000accurate

if __name__ == '__main__':

	ra  = array([x['ra'] for x in ngc2000])
	dec = array([x['dec'] for x in ngc2000])
	radius = array([x['size'] for x in ngc2000])
	ngcnum = array([x['id'] for x in ngc2000])
	# turn from diameter in arcmin to radius in deg.
	radius /= (2. * 60.)
	isngc = array([x['is_ngc'] for x in ngc2000])

	print 'got %i RA (%i ngc)' % (len(ra), sum(isngc))

	prim = pyfits.PrimaryHDU()
	#phdr = prim.header
	#phdr.update('AN_FILE', 'RDLS', 'Astrometry.net RA,Dec list')
	table = pyfits.new_table(
		[pyfits.Column(name='NGCNUM', format='1D', array=ngcnum[isngc], unit=''),
		 pyfits.Column(name='RA', format='1D', array=ra[isngc], unit='deg'),
		 pyfits.Column(name='DEC', format='1D', array=dec[isngc], unit='deg'),
		 pyfits.Column(name='RADIUS', format='1D', array=radius[isngc], unit='deg'),
		 ])
	pyfits.HDUList([prim, table]).writeto('ngc.fits', clobber=True)

	isic = logical_not(isngc)

	prim = pyfits.PrimaryHDU()
	#phdr = prim.header
	#phdr.update('AN_FILE', 'RDLS', 'Astrometry.net RA,Dec list')
	table = pyfits.new_table(
		[pyfits.Column(name='ICNUM', format='1D', array=ngcnum[isic], unit=''),
		 pyfits.Column(name='RA', format='1D', array=ra[isic], unit='deg'),
		 pyfits.Column(name='DEC', format='1D', array=dec[isic], unit='deg'),
		 pyfits.Column(name='RADIUS', format='1D', array=radius[isic], unit='deg'),
		 ])
	pyfits.HDUList([prim, table]).writeto('ic.fits', clobber=True)
