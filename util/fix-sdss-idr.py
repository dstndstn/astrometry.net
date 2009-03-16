import pyfits

# Takes a pyfits object and modifies it.
def fix_sdss_idr(fits):
	hdr = fits[0].header
	if hdr['SIMPLE'] is False:
		hdr.update('SIMPLE', True, 'FITS compliant (fix-sdss-idr.py)')

	if hdr.get('SDSS', None) == pyfits.UNDEFINED:
		hdr['SDSS'] = True
		print 'Set SDSS to True'

	print 'UNSIGNED in hdr?', ('UNSIGNED' in hdr)
	print hdr.get('UNSIGNED', None)

	print hdr.ascardlist()[6]
	print type(hdr.ascardlist()[6])
	print hdr.ascardlist()[6].key
	print hdr.ascardlist()[6].value
	print hdr.ascardlist()[6].comment
	print hdr[6]

	if hdr.get('UNSIGNED', None) == pyfits.UNDEFINED:
		hdr['UNSIGNED'] = True
		print 'Set UNSIGNED to True'
		if (not 'BSCALE' in hdr) and (not 'BZERO' in hdr):
			print 'Adding BZERO = 32768 card'
			hdr.update('BZERO', 32768, 'Unsigned -> signed')

	# ?? What if it *doesn't* contain SDSS and UNSIGNED ??

	print 'type:', type(fits[0])
	print 'data type:', type(fits[0].data)
	print 'hdu type:', fits[0].header._hdutype
	print 'card 0:', fits[0].header.ascard[0]

	fits[0].header._updateHDUtype()

	print 'hdu type:', fits[0].header._hdutype
	data = fits[0].data
	print 'data type:', type(data)
	print 'data:', data

	hdr = fits[0].header
	newhdu = hdr._hdutype(data=data, header=hdr)
	print 'newhdu:', newhdu
	newhdu._file = fits[0]._file
	newhdu._ffile = fits[0]._ffile
	newhdu._datLoc = fits[0]._datLoc
	newhdu.__getattr__('data')
	print 'data:', type(newhdu.data)
	print 'data:', newhdu.data.dtype
	print 'data range:', newhdu.data.min(), 'to', newhdu.data.max()

	newhdu.writeto('out.fits', clobber=True)

# _File._readHDU
#  -> _TempHDU()
#  -> HDUList.__getitem__
#    -> _TempHDU.setupHDU()
#       -> Header(...)
#         -> sets header._hdutype
#       -> header._hdutype(data=...)

# SIMPLE=F -> _ValidHDU


X = fix_sdss_idr(pyfits.open('idR-002987-r3-0059.fit'))
