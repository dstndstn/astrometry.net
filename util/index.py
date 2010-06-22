from index_c import *
from _index_util import *

def index_get_codes(I):
	addr = codekd_addr(I)
	return codekd_get_codes_numpy(addr)

