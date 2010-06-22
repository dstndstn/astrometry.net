import matplotlib
matplotlib.use('Agg')

from astrometry.util.index import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import match
from astrometry.util.starutil_numpy import *

from optparse import *

from pylab import *
from numpy import *

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option('-p', '--prefix', dest='prefix', help='Prefix for output plot names')
	parser.set_defaults(prefix='')
	opt,args = parser.parse_args()

	for indfn in args:
		print 'Reading index', indfn
		null = None
		I = index_load(indfn, 0, null)
		print 'Loaded.'
		NS = index_nstars(I)
		NQ = index_nquads(I)
		print 'Index has', NS, 'stars and', NQ, 'quads'
		DQ = index_get_quad_dim(I)
		print 'Index has "quads" with %i stars' % (DQ)
		DC = index_get_quad_dim(I)
		print 'Index has %i-dimensional codes' % (DC)

		# codes
		print 'Getting codes...'
		codes = zeros((NQ, DQ))
		code = code_alloc(DC)
		for i in range(codetree_N(I.codekd)):
			if codetree_get(I.codekd, i, code):
				raise 'Failed to get code %i' % i
			#print 'code:', code
			for j in range(DQ):
				codes[i,j] = code_get(code, j) #code[j]
		code_free(code);
		#codetree_get_N(I.codekd, 0, NQ, codes.data)
		print 'Codes:', codes.shape

		# code slices
		cx = codes[:,0]
		cy = codes[:,1]
		dx = codes[:,2]
		dy = codes[:,3]
		clf()
		(H,xe,ye) = histogram2d(cx, cy, bins=(100,100))
		H=H.T
		imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
			   interpolation='nearest', origin='lower', cmap=antigray)
		axis('equal')
		xlabel('cx')
		ylabel('cy')
		savefig(opt.prefix + 'codes-1.png')

		clf()
		(H,xe,ye) = histogram2d(append(cx, dx), append(cy, dy), bins=(100,100))
		H=H.T
		imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
			   interpolation='nearest', origin='lower', cmap=antigray)
		axis('equal')
		xlabel('cx, dx')
		ylabel('cy, dy')
		savefig(opt.prefix + 'codes-2.png')

		clf()
		(H,xe,ye) = histogram2d(cx, dx, bins=(100,100))
		H=H.T
		imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
			   interpolation='nearest', origin='lower', cmap=antigray)
		axis('equal')
		xlabel('cx')
		ylabel('dx')
		savefig(opt.prefix + 'codes-3.png')

		clf()
		xx = append(cx, dx)
		yy = append(cy, dy)
		(H,xe,ye) = histogram2d(append(xx, 1.0-xx), append(yy, 1.0-yy), bins=(100,100))
		H=H.T
		imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto',
			   interpolation='nearest', origin='lower', cmap=antigray)
		axis('equal')
		xlabel('cx, dx')
		ylabel('cy, dy')
		title('duplicated for A-B swap')
		savefig(opt.prefix + 'codes-4.png')

		# stars
		print 'Getting stars...'
		stars = zeros((NS, 3))
		# HACK -- abuse code_{alloc,get,free}
		star = code_alloc(3)
		for i in range(startree_N(I.starkd)):
			if startree_get(I.starkd, i, star):
				raise 'Failed to get star %i' % i
			for j in range(3):
				stars[i,j] = code_get(star, j)
		code_free(star)
		print 'Stars:', stars.shape

		R = 15.
		print 'Finding pairs within', R, 'arcsec'
		inds,dists = match(stars, stars, deg2rad(R/3600.))
		print 'inds', inds.shape, 'dists', dists.shape

		notself = (inds[:,0] != inds[:,1])
		clf()
		hist(rad2deg(dists[notself]) * 3600., 200)
		xlabel('Star pair distances (arcsec)')
		ylabel('Counts')
		savefig(opt.prefix + 'stars-1.png')



		index_free(I)
		
