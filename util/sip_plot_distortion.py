import matplotlib
matplotlib.use('Agg')

import sys
from optparse import *

from pylab import *
from numpy import *
from astrometry.util.sip import *

def plot_distortions(wcsfn, ex=1, ngridx=10, ngridy=10, stepx=10, stepy=10):
	wcs = Sip(filename=wcsfn)
	W,H = wcs.wcstan.imagew, wcs.wcstan.imageh

	sx = W / float(ngridx)
	sy = H / float(ngridy)
	xgrid = arange(sx/2., W, sx)
	ygrid = arange(sy/2., H, sy)

	margin = 5
	X = arange(-margin*stepx, W+margin*stepx, stepx)
	Y = arange(-margin*stepy, H+margin*stepy, stepy)

	for x in xgrid:
		DX,DY = [],[]
		xx,yy = [],[]
		for y in Y:
			dx,dy = wcs.get_distortion(x, y)
			xx.append(x)
			yy.append(y)
			DX.append(dx)
			DY.append(dy)
		DX = array(DX)
		DY = array(DY)
		xx = array(xx)
		yy = array(yy)
		EX = DX + ex * (DX - xx)
		EY = DY + ex * (DY - yy)
		plot(xx, yy, 'k-', alpha=0.5)
		plot(EX, EY, 'r-')

	for y in ygrid:
		DX,DY = [],[]
		xx,yy = [],[]
		for x in X:
			dx,dy = wcs.get_distortion(x, y)
			DX.append(dx)
			DY.append(dy)
			xx.append(x)
			yy.append(y)
		DX = array(DX)
		DY = array(DY)
		xx = array(xx)
		yy = array(yy)
		EX = DX + ex * (DX - xx)
		EY = DY + ex * (DY - yy)
		plot(xx, yy, 'k-', alpha=0.5)
		plot(EX, EY, 'r-')

	plot([wcs.wcstan.crpix[0]], [wcs.wcstan.crpix[1]], 'rx')

	axis([0, W, 0, H])


if __name__ == '__main__':
	parser = OptionParser(usage='%prog [options] <wcs-filename> <plot-filename>')
	parser.add_option('-e', '--ex', '--exaggerate', dest='ex', type='float', help='Exaggerate the distortion by this factor')
	#parser.add_option('-s', '--scale', dest='scale', type='float', help='Scale the 
	parser.set_defaults(ex=1.)

	opt,args = parser.parse_args()

	if len(args) != 2:
		parser.print_help()
		sys.exit(-1)

	wcsfn = args[0]
	outfn = args[1]

	clf()
	plot_distortions(wcsfn, opt.ex)
	tt = 'SIP distortions: %s' % wcsfn
	if opt.ex != 1:
		tt += ' (exaggerated by %g)' % opt.ex
	title(tt)
	savefig(outfn)
	
