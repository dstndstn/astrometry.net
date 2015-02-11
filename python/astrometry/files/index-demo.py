#! /usr/bin/env python
import sys
from astrometry.util.index import *

if __name__ == '__main__':

	# ~/deblend/tsObj-000745-3-40-0564.fit

	# build-index -i MyTable_dstn.fit -o sdss.index -N 880  -l 4 -u 5.6 -S ra -r 8 -j 1 -p 16 -R 8 -L 20 -E -I 9999 -M
	#(xyz,radec,inds,tag) = index_search_stars(index, -143, -2, 1, True)

	fn = sys.argv[1]

	index = index_load(fn, 0, None)

	#use_numpy = True
	use_numpy = False

	#(xyz,radec,inds,tag) = index_search_stars(index, -143, -2, 1, True)
	#(xyz,radec,inds,tag) = index_search_stars(index, -117, 0, 1, True, use_numpy)
	(xyz,radec,inds,tag) = index_search_stars(index, -117, 0, 0.1, True, use_numpy)

	if use_numpy:
		print 'Got xyz', xyz.shape
		print 'Got inds', inds.shape

	print inds
	print radec
	for k,v in tag.items():
		print '  ', k, '=', v
	#print tag

