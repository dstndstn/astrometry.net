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
		
	if hdr.get('UNSIGNED', None) == pyfits.UNDEFINED:
		print 'Setting UNSIGNED = True'
		hdr['UNSIGNED'] = True
	else:
		print 'No UNSIGNED header card: not an SDSS idR file.'
		return hdu

	hdr._updateHDUtype()
	newhdu = hdr._hdutype(data=pyfits.DELAYED, header=hdr)
	## HACK - encapsulation violation
	newhdu._file = hdu._file
	newhdu._ffile = hdu._ffile
	newhdu._datLoc = hdu._datLoc

	newhdu.data = newhdu.data.astype(int)
	newhdu.data[newhdu.data < 0] += 2**16
	print 'data type:', newhdu.data.dtype
	print 'data range:', newhdu.data.min(), 'to', newhdu.data.max()
	return newhdu


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'Usage: %s <input-fits-file> <output-fits-file>' % sys.argv[0]
		sys.exit(-1)
	infile = sys.argv[1]
	outfile = sys.argv[2]

	print 'Reading', infile
	newhdu = fix_sdss_idr(pyfits.open(infile)[0])
	print 'Writing', outfile
	newhdu.writeto(outfile, clobber=True)

