# Copyright 2010 Dustin Lang and David W. Hogg, all rights reserved.

# Note: not thoroughly tested.

# http://www.sdss.org/dr7/algorithms/fluxcal.html


from astrometry.util.pyfits_utils import *
from numpy import *
import numpy as np
import pyfits

class PhotometricCalib(object):
	def __init__(self, tsfieldfn):
		# definition of exptime, according to web page above.
		tsfield = fits_table(tsfieldfn)[0]
		self.exptime = 53.907456
		self.aa = tsfield.aa.astype(float)
		self.kk = tsfield.kk.astype(float)
		self.airmass = tsfield.airmass

	
def sdss_maggies_to_mag(flux):
	return -2.5 * np.log10(flux)

def sdss_counts_to_maggies(counts, band, calib):
	return counts/calib.exptime * 10.**(0.4*(calib.aa[band] + calib.kk[band] * calib.airmass[band]))

def sdss_counts_to_mag(counts, band, calib):
	return sdss_maggies_to_mag(sdss_counts_to_maggies(counts, band, calib))

def sdss_mag_to_counts(mag, band, calib):
	logcounts = -0.4 * mag + np.log10(calib.exptime) - 0.4*(calib.aa[band] + calib.kk[band] * calib.airmass[band])
	return 10.**logcounts

if __name__ == '__main__':
	from glob import glob
	fn = glob('tsField-002830-6-*-0398.fit')[0]
	calib = PhotometricCalib(fn)

	tsobj = fits_table(glob('tsObj-002830-6-*-0398.fit')[0])
	fpobj = fits_table('fpObjc-002830-6-0398.fit')

	for band in range(5):
		counts = fpobj.psfcounts[:,band]
		I = counts > 10.**(3.5)
		counts = counts[I].astype(float)
		print 'Counts', counts
		mag = sdss_counts_to_mag(counts, band, calib)
		print 'Mag', mag
		mag2 = tsobj.psfcounts[:, band].astype(float)
		print 'Mag2', mag2
		counts2 = sdss_mag_to_counts(mag, band, calib)
		print 'Counts2', counts2
		for c1,c2 in zip(counts, counts2):
			print c1,c2
	sys.exit(0)

	from pylab import *

	for band in range(5):
		counts = fpobj.psfcounts[:,band]
		I = counts > 10.**(3.5)
		counts = counts[I].astype(float)
		mag = sdss_counts_to_mag(counts, band, calib)
		print mag
		mag2 = tsobj.psfcounts[:, band].astype(float)

		clf()
		dig = (mag2 * 1e4).astype(int) % 10
		hist(dig, bins=range(10))
		savefig('dig1-%i.png' % band)

		clf()
		J = mag2 > -9999
		plot(mag2[J], dig[J], 'r.')
		savefig('dig3-%i.png' % band)

		mag2 = mag2[I]
		clf()
		dig = (mag2 * 1e4).astype(int) % 10
		hist(dig, bins=range(10))
		savefig('dig2-%i.png' % band)

		print mag2

		

		clf()
		axhline(0, linestyle='-', color='k')
		(dmlo,dmhi) = (-1e-3, 1e-3)
		semilogx(counts, clip(mag2-mag, dmlo, dmhi), 'r.')
		xlabel('counts')
		ylabel('dmag')
		ylim(dmlo, dmhi)
		xlim(min(counts), max(counts))
		savefig('dmag-%i.png' % band)

		clf()

		subplot(2,1,1)
		axhline(0, linestyle='-', color='k')
		(dmlo,dmhi) = (-1e-3, 1e-3)
		plot(fpobj.rowc[I,band], clip(mag2-mag, dmlo, dmhi), 'r.')
		xlabel('rowc')
		ylabel('dmag')
		ylim(dmlo, dmhi)

		subplot(2,1,2)
		axhline(0, linestyle='-', color='k')
		(dmlo,dmhi) = (-1e-3, 1e-3)
		plot(fpobj.colc[I,band], clip(mag2-mag, dmlo, dmhi), 'r.')
		xlabel('colc')
		ylabel('dmag')
		ylim(dmlo, dmhi)

		savefig('dmagxy-%i.png' % band)

