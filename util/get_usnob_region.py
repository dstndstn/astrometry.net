from urllib2 import urlopen
from urllib import urlencode
from urlparse import urlparse, urljoin

from numpy import *

from get_usnob import *
from astrometry.util.file import *

from optparse import OptionParser


if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option('-r', '--ra-low', '--ra-lo', '--ra-min',
					  dest='ralo', type=float, help='Minimum RA')
	parser.add_option('-R', '--ra-high', '--ra-hi', '--ra-max',
					  dest='rahi', type=float, help='Maximum RA')
	parser.add_option('-d', '--dec-low', '--dec-lo', '--dec-min',
					  dest='declo', type=float, help='Minimum Dec')
	parser.add_option('-D', '--dec-high', '--dec-hi', '--dec-max',
					  dest='dechi', type=float, help='Maximum Dec')
	parser.add_option('-p', '--prefix',
					  dest='prefix', help='Output file prefix')
	parser.add_option('-s', '--survey',
					  dest='survey', help='Grab only one USNOB survey: poss-i, poss-ii, ... (see http://www.nofs.navy.mil/data/fchpix/cfch.html')
	parser.add_option('-P', '--plate',
					  dest='plate', help='Grab only one USNOB plate: "se0161", for example')

	parser.set_defaults(prefix='usnob', survey=None, plate=None,
						ralo=None, rahi=None, declo=None, dechi=None)
	(opt, args) = parser.parse_args()

	if opt.ralo is None or opt.rahi is None or opt.declo is None or opt.dechi is None:
		parser.print_help()
		parser.error('RA,Dec lo,hi are required.')

	decstep = 14./60.
	Dec = arange(opt.declo, opt.dechi+decstep, decstep)
	for dec in Dec:
		rastep = 14./60./cos(deg2rad(dec))
		RA  = arange(opt.ralo , opt.rahi +rastep , rastep)
		for ra in RA:
			(jpeg,fits) = get_usnob_images(ra, dec, fits=True, survey=opt.survey, justurls=True)
			print 'got jpeg urls:', jpeg
			print 'got fits urls:', fits
			if opt.plate is None:
				keepjpeg = jpeg
				keepfits = fits
			else:
				keepjpeg = [u for u in jpeg if opt.plate in u]
				keepfits = [u for u in fits if opt.plate in u]
				print 'keep jpeg urls:', keepjpeg
				print 'keep fits urls:', keepfits
			base = opt.prefix + '-%.3f-%.3f-' % (ra,dec)
			for url in keepjpeg:
				fn = base + url.split('/')[-1]
				print 'retrieving', url, 'to', fn
				res = urlopen(url)
				write_file(res.read(), fn)
			for url in keepfits:
				fn = base + url.split('/')[-1] + '.fits'
				print 'retrieving', url, 'to', fn
				res = urlopen(url)
				write_file(res.read(), fn)

		




def old():
	#prefix = 'mosaic'
	#(ralo,  rahi ) = (161.6, 165.2)
	#(declo, dechi) = ( 43.0,  45.5)

	#prefix = 'mosaic-c-'
	#(ralo,  rahi ) = (162.75, 165.25)
	#(declo, dechi) = ( 44.3,   44.29)

	#prefix = 'mosaic-d'
	#(ralo,  rahi ) = (162.75, 162.74)
	#(declo, dechi) = ( 43.,    44.3 )

	#survey = 'poss-i'
	#plate = 'se0215'

	#prefix = 'mosaic-e'
	#(ralo,  rahi ) = (95.0, 100.2)
	#(declo, dechi) = (49.5,  52.6)

	prefix = 'mosaic-f'
	(ralo,  rahi ) = (95.0, 100.2)
	(declo, dechi) = (51.0, 50.99)

	survey = 'poss-i'
	plate = 'se0161'

	rastep = 20./60.  # (14 / cos(deg2rad(45)))
	decstep = 14./60.

	RA  = arange(ralo , rahi +rastep , rastep)
	Dec = arange(declo, dechi+decstep, decstep)

	print 'RA:', RA
	print 'Dec:', Dec

	for dec in Dec:
		for ra in RA:
			(jpeg,fits) = get_usnob_images(ra, dec, fits=True, survey=survey, justurls=True)
			print 'got jpeg urls:', jpeg
			print 'got fits urls:', fits
			keepjpeg = [u for u in jpeg if plate in u]
			keepfits = [u for u in fits if plate in u]
			print 'keep jpeg urls:', keepjpeg
			print 'keep fits urls:', keepfits

			base = prefix + '-%.3f-%.3f-' % (ra,dec)

			for url in keepjpeg:
				fn = base + url.split('/')[-1]
				print 'retrieving', url, 'to', fn
				res = urlopen(url)
				write_file(res.read(), fn)
		
			for url in keepfits:
				fn = base + url.split('/')[-1] + '.fits'
				print 'retrieving', url, 'to', fn
				res = urlopen(url)
				write_file(res.read(), fn)

