import pyfits
from optparse import OptionParser
from astrometry.util.pyfits_utils import *

if __name__ == '__main__':
	parser = OptionParser(usage='%prog [options] <match-input-file> <wcs-output-file>')

	parser.add_option('-r', dest='row', help='Row of the match file to take (default 0)', type='int')
	parser.add_option('-W', dest='imgw', help='Image width', type='float')
	parser.add_option('-H', dest='imgh', help='Image height', type='float')

	parser.set_defaults(row=0, imgw=0, imgh=0)

	(options, args) = parser.parse_args()
	if len(args) != 2:
		parser.print_help()
		print
		print 'Need args <match-input-file> and <wcs-output-file>'
		sys.exit(-1)

	matchfn = args[0]
	wcsfn = args[1]


	m = table_fields(matchfn)
	I = options.row
	
	wcs = pyfits.PrimaryHDU()
	h = wcs.header

	hdrs = [
		('CTYPE1', 'RA---TAN', None),
		('CTYPE2', 'DEC--TAN', None),
		('WCSAXES', 2, None),
		('EQUINOX', 2000.0, 'Equatorial coordinates definition (yr)'),
		('LONPOLE', 180.0, None),
		('LATPOLE', 0.0, None),
		('CUNIT1', 'deg', 'X pixel scale units'),
		('CUNIT2', 'deg', 'Y pixel scale units'),
		('CRVAL1', m.crval[I][0], 'RA  of reference point'),
		('CRVAL2', m.crval[I][1], 'DEC of reference point'),
		('CRPIX1', m.crpix[I][0], 'X reference pixel'),
		('CRPIX2', m.crpix[I][1], 'Y reference pixel'),
		('CD1_1', m.cd[I][0], 'Transformation matrix'),
		('CD1_2', m.cd[I][1], None),
		('CD2_1', m.cd[I][2], None),
		('CD2_2', m.cd[I][3], None),
		]
	if options.imgw:
		hdrs.append(('IMAGEW', options.imgw, 'Image width,  in pixels.'))
	if options.imgh:
		hdrs.append(('IMAGEH', options.imgh, 'Image height, in pixels.'))

	for (k,v,c) in hdrs:
		h.update(k, v, c)
	wcs.writeto(wcsfn, clobber=True)

