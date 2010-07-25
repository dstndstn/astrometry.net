#! /usr/bin/env python
import sys
from astrometry.util.index import *

if __name__ == '__main__':

	# ~/deblend/tsObj-000745-3-40-0564.fit

	# build-index -i MyTable_dstn.fit -o sdss.index -N 880  -l 4 -u 5.6 -S ra -r 8 -j 1 -p 16 -R 8 -L 20 -E -I 9999 -M
	#(xyz,radec,inds,tag) = index_search_stars(index, -143, -2, 1, True)

	fn = sys.argv[1]

	index = index_load(fn, 0, None)

	#(xyz,radec,inds,tag) = index_search_stars(index, -143, -2, 1, True)
	(xyz,radec,inds,tag) = index_search_stars(index, -117, 0, 1, True)
	print 'Got xyz', xyz.shape
	print 'Got inds', inds.shape
	print inds
	print tag
	#print 'tag-along cols:', index_get_tagalong_columns(index)
