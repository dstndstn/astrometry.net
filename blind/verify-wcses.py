from astrometry.util.healpix import *
from astrometry.util.pyfits_utils import *

from numpy import *
from pyfits import *

import ctypes
import ctypes.util

_lib = None
_libname = ctypes.util.find_library('libbackend.so')
if _libname:
	_lib = ctypes.CDLL(_libname)
if _lib is None:
	p = os.path.join(os.path.dirname(__file__), 'libbackend.so')
	if os.path.exists(p):
		_lib = ctypes.CDLL(p)
if _lib is None:
	raise IOError('libbackend.so library not found')

def load_startree(fn):
	ptr = _lib.startree_open(create_string_buffer(fn))
	return ptr

def starxy_from_arrays(x, y):
	N = len(x)
	assert(len(y) == len(x))
	cfalse = c_char('\0')
	s = _lib.starxy_new(c_int(N), cfalse, cfalse);
	for i,(xi,yi) in enumerate(zip(x, y)):
		_lib.starxy_set_x(c_void_p(s), c_int(i), c_double(xi))
		_lib.starxy_set_y(c_void_p(s), c_int(i), c_double(yi))
	return s

def starxy_free(s):
	_lib.starxy_free(s)

def verify_field_preprocess(s):
	return _lib.verify_field_preprocess(s)

def verify_field_free(vf):
	return _lib.verify_field_free(vf)

def sip_read_header(fn):
	return _lib.sip_read_header_file(create_string_buffer(fn), c_void_p(None))

def sip_free(s):
	return _lib.free(s)

def verify_wcs(starkd, indexcutnside, sip, vf,
			   verify_pix2, distractors, imagew, imageh,
			   logbail, logaccept, logstop):

	logodds = c_double(0)
	null = c_void_p(None)
	_lib.verify_wcs(starkd, c_int(indexcutnside), sip, vf, c_double(verify_pix2),
					c_double(distractors), c_double(imagew), c_double(imageh),
					c_double(logbail), c_double(logaccept), c_double(logstop),
					pointer(logodds), null, null, null, null, null)
	return float(logodds.value)

def an_lib_setup():
	# 2 = normal, 3 = verbose, 4 = debug
	_lib.log_init(3)

if __name__ == '__main__':

	an_lib_setup()

	hps = [4]
	indexfns = ['/Users/dstn/INDEXES/index-604-%02i.fits' % hp for hp in hps]
	starkds = [load_startree(fn) for fn in indexfns]

	print 'got starkdtrees:', starkds

	xylistfns = ['testCat.fits']
	wcsfns = ['PE00050.063.fits']
	solvedlogodds = log(1e9)

	lodds = []

	for xyfn,wcsfn in zip(xylistfns, wcsfns):
		xy = table_fields(xyfn)
		I = argsort(xy.mag_auto)
		starx = xy.xwin_image[I]
		stary = xy.ywin_image[I]

		#print zip(starx,stary)[:10]

		starxy = starxy_from_arrays(starx, stary)
		vf = verify_field_preprocess(starxy)
		sip = sip_read_header(wcsfn)

		print 'read xylist', xyfn, 'and wcs', wcsfn

		# HACK
		indexcutnside = 2
		imagew, imageh = (1394, 1037)

		# HACK - find appropriate healpixes...
		for starkd in starkds:
			print 'verifying...'
			logodds = verify_wcs(starkd, indexcutnside, sip, vf,
								 1, 0.25, imagew, imageh,
								 log(1e-100), log(1e9), log(1e300))
			print 'logodds', logodds
			lodds.append(logodds)

		sip_free(sip)
		verify_field_free(vf)
		starxy_free(starxy)

	hdu = pyfits.new_table([
		pyfits.Column(name='sourcefn', format='32A', array=xylistfns),
		pyfits.Column(name='wcsfn', format='32A', array=wcsfns),
		pyfits.Column(name='logodds', format='E', array=lodds),
		pyfits.Column(name='solved', format='L', array=(lodds > solvedlogodds)),
		])
	hdu.writeto('logodds.fits', clobber=True)
	
	#healpix.healpix_neighbours_within_range

	# -load set of star kdtree files
	# -for each wcs,xylist:
	#   -find star kdtrees within range
	#    (using star kdtree healpix,nside, and wcs crval)
	#   -verify wcs,xylist against each star kdtree
	#   -take max logodds

