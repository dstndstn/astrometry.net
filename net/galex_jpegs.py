import os

from astrometry.util.pyfits_utils import fits_table
from astrometry.util.util import Tan
from astrometry.blind import plotstuff as ps
#from astrometry.libkd.spherematch import *

from astrometry.util.starutil_numpy import *

import numpy as np

def plot_into_wcs(wcsfn, plotfn, wcsext=0, basedir='.'):
	wcs = Tan(wcsfn, wcsext)
	ra,dec = wcs.radec_center()
	radius = wcs.radius()

	T = fits_table(os.path.join(basedir, 'index.fits'))
	r = sqrt(radius**2 + 1.**2)

	# search... this is probably more firepower than we need...
	#I,J,d = match_radec(np.array([ra]), np.array([dec]), T.ra, T.dec, r)
	#print len(I), 'matches'
	#r2 = deg2distsq(r)
	#xyz = radectoxyz(

	J = points_within_radius(ra, dec, r, T.ra, T.dec)
	T = T[J]
	print len(T), 'matches'

	plot = ps.Plotstuff(outformat='png', wcsfn=wcsfn, wcsext=wcsext)
	img = plot.image
	img.format = ps.PLOTSTUFF_FORMAT_JPG
	img.resample = 1
	for jpegfn in T.path:
		jpegfn = os.path.join(basedir, jpegfn)
		imwcsfn = jpegfn.replace('.jpg', '.wcs')
		print 'Plotting', jpegfn, imwcsfn
		img.set_wcs_file(imwcsfn, 0)
		img.set_file(jpegfn)
		# convert black to transparent
		ps.plot_image_read(plot.pargs, img)
		ps.plot_image_make_color_transparent(img, 0, 0, 0)
		plot.plot('image')
	plot.write(plotfn)

