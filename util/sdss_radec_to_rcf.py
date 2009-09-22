import pyfits
from astrometry.util.pyfits_utils import *
from astrometry.util.starutil_numpy import *
from astrometry.util.find_data_file import *
from os.path import basename,dirname

# RA,Dec are either scalars or iterables.
# If scalars, returns a list of (run, camcol, field, ra, dec) tuples, one for each matching field.
# If iterable, returns a list containing one list per query (ra,dec) of the same tuple.
def radec_to_sdss_rcf(ra, dec, spherematch=True):
	fn = find_data_file('dr7fields.fits')
	sdss = table_fields(fn)
	sdssxyz = radectoxyz(sdss.ra, sdss.dec)
	## HACK - magic 13x9 arcmin.
	radius2 = arcmin2distsq(sqrt(13.**2 + 9.**2)/2.)
	if not spherematch:
		rcfs = []
		for r,d in broadcast(ra,dec):
			xyz = radectoxyz(r,d)
			dist2s = sum((xyz - sdssxyz)**2, axis=1)
			I = flatnonzero(dist2s < radius2)
			if False:
				print 'I:', I
				print 'fields:', sdss[I].run, sdss[I].field, sdss[I].camcol
				print 'RA', sdss[I].ra
				print 'Dec', sdss[I].dec
			rcfs.append(zip(sdss[I].run, sdss[I].camcol, sdss[I].field, sdss[I].ra, sdss[I].dec))
	else:
		from astrometry.libkd import spherematch
		rds = array([x for x in broadcast(ra,dec)])
		#print 'rds shape:', rds.shape
		xyz = radectoxyz(rds[:,0], rds[:,1]).astype(double)
		#print 'xyz shape:', xyz.shape
		#print 'xyz type:', xyz.dtype
		#print 'sdss xyz shape:', sdssxyz.shape
		#print 'sdss type:', sdssxyz.dtype
		(inds,dists) = spherematch.match(xyz, sdssxyz, sqrt(radius2))
		#print 'found %i matches' % len(inds)
		#print 'inds:', inds.shape
		rcfs = [[] for i in range(len(rds))]
		#print 'len rds:', len(rds)
		#print 'len sdssxyz:', len(sdssxyz)
		#print 'len sdss:', len(sdss)
		#print 'len rcfs:', len(rcfs)
		for i,j in inds:
			#print 'inds', i, j
			#print '  NGC ra,dec', rds[i,:]
			#print '  SDSS ra,dec', sdss.ra[j], sdss.dec[j]
			#print '  rcfs:', rcfs[i]
			#print '  adding:', (sdss.run[j], sdss.camcol[j], sdss.field[j], sdss.ra[j], sdss.dec[j])
			rcfs[i].append((sdss.run[j], sdss.camcol[j], sdss.field[j], sdss.ra[j], sdss.dec[j]))

	if isscalar(ra) and isscalar(dec):
		return rcfs[0]
	return rcfs

# The field list was created starting with dstn's list of fields in DR7:
#  fitscopy dr7_e.fits"[col RUN;FIELD;CAMCOL;RA=(RAMIN+RAMAX)/2;DEC=(DECMIN+DECMAX)/2]" e.fits
#  fitscopy dr7_g.fits"[col RUN;FIELD;CAMCOL;RA=(RAMIN+RAMAX)/2;DEC=(DECMIN+DECMAX)/2]" g.fits
#  fitscopy dr7_a.fits"[col RUN;FIELD;CAMCOL;RA=(RAMIN+RAMAX)/2;DEC=(DECMIN+DECMAX)/2]" a.fits
#  tabmerge g.fits e.fits
#  tabmerge g.fits+1 e.fits+1
#  tabmerge a.fits+1 e.fits+1
#  mv e.fits dr7fields.fits
#  rm g.fits a.fits

if __name__ == '__main__':
	rcfs = radec_to_sdss_rcf([236.1, 236.4], [0,0])
	print 'rcfs:', rcfs

