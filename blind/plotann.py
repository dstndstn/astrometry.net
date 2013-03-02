#! /usr/bin/env python
import sys
import os

# from util/addpath.py
if __name__ == '__main__':
	try:
		import astrometry
		from astrometry.util.shell import shell_escape
		from astrometry.util.filetype import filetype_short
	except ImportError:
		me = __file__
		path = os.path.realpath(me)
		blinddir = os.path.dirname(path)
		assert(os.path.basename(blinddir) == 'blind')
		andir = os.path.dirname(blinddir)
		if os.path.basename(andir) == 'astrometry':
			rootdir = os.path.dirname(andir)
			sys.path.insert(1, andir)
		else:
			# assume there's a symlink astrometry -> .
			rootdir = andir
		#sys.path += [rootdir]
		sys.path.insert(1, rootdir)

from optparse import OptionParser

from astrometry.blind.plotstuff import *
from astrometry.util.fits import *

if __name__ == '__main__':
	parser = OptionParser('usage: %prog <wcs.fits file> <image file> <output.{jpg,png,pdf} file>')
	parser.add_option('--scale', dest='scale', type=float,
					  help='Scale plot by this factor')
	parser.add_option('--hdcat', dest='hdcat',
					  help='Path to Henry Draper catalog hd.fits')
	parser.add_option('--uzccat', dest='uzccat',
					  help='Path to Updated Zwicky Catalog uzc2000.fits')
	parser.add_option('--abellcat', dest='abellcat',
					  help='Path to Abell catalog abell-all.fits')
	parser.add_option('--target', '-t', dest='target', action='append',
					  default=[],
					  help='Add named target (eg "M 31", "NGC 1499")')
	parser.add_option('--no-grid', dest='grid', action='store_false',
					  default=True, help='Turn off grid lines')

	parser.add_option('--tcolor', dest='textcolor', default='green',
					  help='Text color')
	parser.add_option('--tsize', dest='textsize', default=18, type=float,
					  help='Text font size')
	parser.add_option('--halign', dest='halign', default='C',
					  help='Text horizontal alignment')
	parser.add_option('--valign', dest='valign', default='B',
					  help='Text vertical alignment')
	parser.add_option('--tox', dest='tox', default=0, type=float,
					  help='Text offset x')
	parser.add_option('--toy', dest='toy', default=0, type=float,
					  help='Text offset y')
	parser.add_option('--lw', dest='lw', default=2, type=float,
					  help='Annotations line width')
	parser.add_option('--ms', dest='ms', default=12, type=float,
					  help='Marker size')
	parser.add_option('--rd', dest='rd', action='append', default=[],
					  help='Plot RA,Dec markers')
	parser.add_option('--quad', dest='quad', action='append', default=[],
					  help='Plot quad from given match file')
	
	opt,args = parser.parse_args()
	if len(args) != 3:
		parser.print_help()
		sys.exit(-1)

	wcsfn = args[0]
	imgfn = args[1]
	outfn = args[2]
	
	fmt = PLOTSTUFF_FORMAT_JPG
	s = outfn.split('.')
	if len(s):
		s = s[-1].lower()
		if s in Plotstuff.format_map:
			fmt = s
	plot = Plotstuff(outformat=fmt, wcsfn=wcsfn)
	#plot.wcs_file = wcsfn
	#plot.outformat = fmt
	#plotstuff_set_size_wcs(plot.pargs)

	plot.outfn = outfn
	img = plot.image
	img.set_file(imgfn)

	if opt.scale:
		plot.scale_wcs(opt.scale)
		plot.set_size_from_wcs()
		#W,H = img.get_size()

	plot.plot('image')

	if opt.grid:
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
	if opt.uzccat:
		# FIXME -- is this fast enough, or do we need to cut these
		# targets first?
		#print >> sys.stderr, 'Plot size', plot.W, plot.H, 'wcs', plot.wcs
		T = fits_table(opt.uzccat)
		for i in range(len(T)):
			if not plot.wcs.is_inside(T.ra[i], T.dec[i]):
				continue
			ann.add_target(T.ra[i], T.dec[i], 'UZC %s' % T.zname[i])
	if opt.abellcat:
		T = fits_table(opt.abellcat)
		for i in range(len(T)):
			if not plot.wcs.is_inside(T.ra[i], T.dec[i]):
				continue
			ann.add_target(T.ra[i], T.dec[i], 'Abell %i' % T.aco[i])
			
	plot.color = opt.textcolor
	plot.fontsize = opt.textsize
	plot.lw = opt.lw
	plot.valign = opt.valign
	plot.halign = opt.halign
	plot.label_offset_x = opt.tox;
	plot.label_offset_y = opt.toy;
	
	if len(opt.target):
		for t in opt.target:
			if plot_annotations_add_named_target(ann, t):
				raise RuntimeError('Unknown target', t)

	plot.plot('annotations')

	for rdfn in opt.rd:
		rd = plot.radec
		rd.fn = rdfn
		plot.markersize = opt.ms
		plot.plot('radec')

	for mfn in opt.quad:
		match = fits_table(mfn)
		for m in match:
			qp = m.quadpix
			xy = [(qp[0], qp[1])]
			#plot.move_to_xy(qp[0], qp[1])
			for d in range(1, m.dimquads):
				#plot.line_to_xy(qp[2 * d], qp[2 * d + 1])
				xy.append((qp[2 * d], qp[2 * d + 1]))
			#plot.stroke()
			plot.polygon(xy)
			plot.close_path()
			plot.stroke()
		
	plot.write()
