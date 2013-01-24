import pyfits
from numpy import *
from ngc2000 import ngc2000, ngc2000accurate
from astrometry.util.pyfits_utils import *

if __name__ == '__main__':

	# convert to a form that leads to simple updating with the "accurate" versions.
	nummap = dict([[(x['is_ngc'], x['id']), (x['ra'],x['dec'],x['size'])]
				   for x in ngc2000])
	print 'got %i' % len(nummap)

	# update with "accurate" values.
	nup = 0
	for x in ngc2000accurate:
		key = (x['is_ngc'], x['id'])
		if key in nummap:
			(oldra, olddec, size) = nummap[key]
			nummap[key] = (x['ra'], x['dec'], size)
			nup +=1
	print 'updated %i' % nup

	isngc  = array([i for (i,n) in nummap.iterkeys()])
	ngcnum = array([n for (i,n) in nummap.iterkeys()])
	ra     = array([r for (r,d,s) in nummap.itervalues()])
	dec    = array([d for (r,d,s) in nummap.itervalues()])
	radius = array([s for (r,d,s) in nummap.itervalues()])
	# turn from diameter in arcmin to radius in deg.
	radius /= (2. * 60.)

	print 'got %i RA (%i ngc)' % (len(ra), sum(isngc))

	prim = pyfits.PrimaryHDU()
	#phdr = prim.header
	#phdr.update('AN_FILE', 'RDLS', 'Astrometry.net RA,Dec list')
	table = pyfits.new_table(
		[pyfits.Column(name='NGCNUM', format='1I', array=ngcnum[isngc], unit=''),
		 pyfits.Column(name='RA', format='1E', array=ra[isngc], unit='deg'),
		 pyfits.Column(name='DEC', format='1E', array=dec[isngc], unit='deg'),
		 pyfits.Column(name='RADIUS', format='1E', array=radius[isngc], unit='deg'),
		 ])
	pyfits_writeto(pyfits.HDUList([prim, table]), 'ngc.fits')

	isic = logical_not(isngc)

	prim = pyfits.PrimaryHDU()
	#phdr = prim.header
	#phdr.update('AN_FILE', 'RDLS', 'Astrometry.net RA,Dec list')
	table = pyfits.new_table(
		[pyfits.Column(name='ICNUM', format='1I', array=ngcnum[isic], unit=''),
		 pyfits.Column(name='RA', format='1E', array=ra[isic], unit='deg'),
		 pyfits.Column(name='DEC', format='1E', array=dec[isic], unit='deg'),
		 pyfits.Column(name='RADIUS', format='1E', array=radius[isic], unit='deg'),
		 ])
	pyfits_writeto(pyfits.HDUList([prim, table]), 'ic.fits')
