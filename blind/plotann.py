import sys
from optparse import OptionParser

from astrometry.blind.plotstuff import *

if __name__ == '__main__':
	parser = OptionParser('usage: %prog <wcs.fits file> <image file> <output.jpg file>')
	parser.add_option('--hdcat', dest='hdcat',
					  help='Path to Henry Draper catalog hd.fits')
	parser.add_option('--target', '-t', dest='target', action='append',
					  default=[],
					  help='Add named target (eg "M 31", "NGC 1499")')
	
	opt,args = parser.parse_args()
	if len(args) != 3:
		parser.print_help()
		sys.exit(-1)

	wcsfn = args[0]
	imgfn = args[1]
	outfn = args[2]
	
	plot = Plotstuff()
	plot.wcs_file = wcsfn
	plot.outformat = PLOTSTUFF_FORMAT_JPG
	plot.outfn = outfn
	plotstuff_set_size_wcs(plot.pargs)
	img = plot.image
	img.set_file(imgfn)
	plot.plot('image')

	plot.color = 'gray'
	plot.plot_grid(0.1, 0.1, 0.2, 0.2)

	ann = plot.annotations
	ann.NGC = True
	ann.constellations = True
	ann.bright = True
	ann.ngc_fraction = 0.
	if opt.hdcat:
		ann.HD = True
		ann.hd_catalog = opt.hdcat
	plot.color = 'green'
	plot.fontsize = 18
	plot.lw = 2.

	if len(opt.target):
		for t in opt.target:
			if plot_annotations_add_named_target(ann, t):
				raise RuntimeError('Unknown target', t)

	plot.plot('annotations')
	plot.write()
