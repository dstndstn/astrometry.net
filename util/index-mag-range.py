import matplotlib
matplotlib.use('Agg')

from astrometry.util.util import *

import numpy as np
import pylab as plt

if __name__ == '__main__':
	indexfn = '/Users/dstn/INDEXES/tycho-2/tycho-2811.index'
	magname = 'MAG'
	
	index = index_load(indexfn, 0, None)
	print index

	print 'Index', indexfn, 'has', index_nstars(index), 'stars and',
	print index_nquads(index), 'quads'

	quads = index.quads
	print 'quads:', quads

	starkd = index.starkd;
	print 'star kdtree:', starkd

	tagalong = startree_get_tagalong(starkd)
	print 'tag-along:', tagalong

	mags = startree_get_data_column(starkd, magname, None, startree_N(starkd))
	print 'mags:', mags[:10], '...'

	print 'mag range:', min(mags), max(mags)

	mn,mx = [],[]
	for i in range(index_nquads(index)):
		stars = quadfile_get_stars(quads, i)
		#print 'stars', stars
		qmags = [mags[j] for j in stars]
		#print 'mags', qmags
		mn.append(min(qmags))
		mx.append(max(qmags))

		
	mn = np.array(mn)
	mx = np.array(mx)
		
	H,xe,ye = np.histogram2d(mn, mx, bins=50)
	plt.clf()
	plt.imshow(np.log10(1+H.T), extent=[min(xe),max(xe),min(ye),max(ye)],
			   origin='lower', interpolation='nearest')
	plt.xlabel('Min mag')
	plt.ylabel('Max mag')
	plt.savefig('magrange.png')


	H,xe,ye = np.histogram2d(mn, mx-mn, bins=50)
	plt.clf()
	plt.imshow(np.log10(1+H.T), extent=[min(xe),max(xe),min(ye),max(ye)],
			   origin='lower', interpolation='nearest')
	plt.xlabel('Min mag')
	plt.ylabel('Mag range')
	plt.savefig('magrange2.png')
