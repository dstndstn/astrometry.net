from index_c import *
from _index_util import *

def index_get_codes(I):
	addr = codekd_addr(I)
	return codekd_get_codes_numpy(addr)

def index_get_stars(I):
	addr = starkd_addr(I)
	return starkd_get_stars_numpy(addr)

# RA, Dec, radius in deg.
# Returns (xyz, radec, inds)
def index_search_stars(I, ra, dec, radius):
	addr = starkd_addr(I)
	return starkd_search_stars_numpy(addr, ra, dec, radius)


