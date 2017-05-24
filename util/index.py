# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from index_c import *
from _index_util import *

def index_get_codes(I):
    addr = codekd_addr(I)
    return codekd_get_codes_numpy(addr)

def index_get_stars(I):
    addr = starkd_addr(I)
    return starkd_get_stars_numpy(addr)

# RA, Dec, radius in deg.
# Returns (xyz, radec, inds[, tagalong])
# "tagalong", if requested, is a dict of column name -> numpy array.
def index_search_stars(I, ra, dec, radius, tagalong=False, numpy=True):
    addr = starkd_addr(I)
    if numpy:
        return starkd_search_stars_numpy(addr, ra, dec, radius, tagalong)
    else:
        return starkd_search_stars(addr, ra, dec, radius, tagalong)

# Returns a list of  (name, fits_type, array_size)
def index_get_tagalong_columns(index):
    skdt = index.starkd
    N = startree_get_tagalong_N_columns(skdt)
    cols = []
    for i in range(N):
        col = startree_get_tagalong_column_name(skdt, i)
        print('column:', col)
        ft = startree_get_tagalong_column_fits_type(skdt, i)
        print('fits type', ft)
        arr = startree_get_tagalong_column_array_size(skdt, i)
        print('array size', arr)
        cols.append((col, ft, arr))
    return cols

#def index_get_tagalong(index, inds):
#    #skdt = index.starkd
#    #addr = starkd_addr(I)
#    #tagalong = {}
#    #for (col,ft,arr) in index_get_tagalong_columns(index):
#    #    startree_get_
#    #return tagalong
#    return starkd_get_tagalong_python(starkd_addr(index), inds)


