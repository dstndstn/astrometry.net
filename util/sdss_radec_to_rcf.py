import pyfits
from astrometry.util.pyfits_utils import *
from astrometry.util.starutil_numpy import *
#from numpy import *

def radec_to_sdss_rcf(ra, dec):
	sdss = table_fields('dr7_e.fits')
	sdssxyz = radectoxyz((sdss.ramin + sdss.ramax)/2.,
						 (sdss.decmin + sdss.decmax)/2.)
	## HACK - magic 13x9 arcmin.
	radius2 = arcmin2distsq(sqrt(13.**2 + 9.**2)/2.)

	rcfs = []
	for r,d in broadcast(ra,dec):
		xyz = radectoxyz(r,d)
		dist2s = sum((xyz - sdssxyz)**2, axis=1)
		I = flatnonzero(dist2s < radius2)
		if False:
			print 'I:', I
			print 'fields:', sdss[I].run, sdss[I].field, sdss[I].camcol
			print 'RA min', sdss[I].ramin
			print 'RA max', sdss[I].ramax
			print 'Dec min', sdss[I].decmin
			print 'Dec max', sdss[I].decmax
		rcfs.append(zip(sdss[I].run, sdss[I].camcol, sdss[I].field))

	return rcfs

if __name__ == '__main__':
	rcfs = radec_to_sdss_rcf([236.1, 236.4], [0,0])
	print 'rcfs:', rcfs
