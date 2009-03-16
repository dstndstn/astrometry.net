import sys
import pyfits
import numpy

# Takes a pyfits HDU object (which is unchanged) and returns a new
# pyfits object containing the fixed file.
def fix_sdss_idr(hdu):
	hdr = hdu.header.copy()

	if hdr.get('SIMPLE', True):
		print 'SIMPLE = T: not an SDSS idR file.'
		return hdu
	print 'Setting SIMPLE = True'
	hdr.update('SIMPLE', True, 'FITS compliant (via fix-sdss-idr.py)')

	if hdr.get('SDSS', None) == pyfits.UNDEFINED:
		print 'Setting SDSS = True'
		hdr['SDSS'] = True
	else:
		print 'No SDSS header card: not an SDSS idR file.'
		return hdu
		
	if (hdr.get('UNSIGNED', None) == pyfits.UNDEFINED
		and (not 'BSCALE' in hdr) and (not 'BZERO' in hdr)):
		print 'Setting UNSIGNED = True'
		hdr['UNSIGNED'] = True
		#print 'Adding BZERO = 32768 card'
		#hdr.update('BZERO', 32768, 'Unsigned -> signed')
	else:
		print 'No UNSIGNED header card, or unexpected BSCALE or BZERO card: not an SDSS idR file.'
		return hdu

	hdr._updateHDUtype()
	newhdu = hdr._hdutype(data=pyfits.DELAYED, header=hdr)
	#print 'newhdu:', newhdu
	## HACK - encapsulation violation
	newhdu._file = hdu._file
	newhdu._ffile = hdu._ffile
	newhdu._datLoc = hdu._datLoc
	#newhdu.__getattr__('data')
	#newhdu.data = newhdu.data.astype(numpy.float32)
	newhdu.data = newhdu.data.astype(int)
	newhdu.data[newhdu.data < 0] += 2**16
	print 'data:', type(newhdu.data)
	print 'data:', newhdu.data.dtype
	print 'data range:', newhdu.data.min(), 'to', newhdu.data.max()

	from pylab import *
	clf()
	hist(newhdu.data.ravel(), 100)
	savefig('hist.png')

	return newhdu

	#newhdu2 = pyfits.PrimaryHDU(data=pyfits.DELAYED, header=hdr)
	#newhdu2 = newhdu2.setupHDU()
	#print 'newhdu2:', newhdu2
	#print 'data:', type(newhdu2.data)
	#print 'data:', newhdu2.data.dtype
	#print 'data range:', newhdu2.data.min(), 'to', newhdu2.data.max()



if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'Usage: %s <input-fits-file> <output-fits-file>' % sys.argv[0]
		sys.exit(-1)
	infile = sys.argv[1]
	outfile = sys.argv[2]

	newhdu = fix_sdss_idr(pyfits.open(infile)[0])
	newhdu.writeto(outfile, clobber=True)

