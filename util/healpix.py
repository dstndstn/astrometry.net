from math import *
from starutil import *

import unittest

import ctypes
import ctypes.util
import os.path
from ctypes import *

_lib = None
_libname = ctypes.util.find_library('_healpix.so')
if _libname:
	_lib = ctypes.CDLL(_libname)
if _lib is None:
	p = os.path.join(os.path.dirname(__file__), '_healpix.so')
	if os.path.exists(p):
		_lib = ctypes.CDLL(p)
if _lib is None:
	raise IOError('_healpix.so library not found')


# returns (base hp, x, y)
def decompose_xy(hp, nside):
	finehp = hp % (nside**2)
	return (int(hp / (nside**2)), int(finehp / nside), finehp % nside)

def get_base_neighbour(hp, dx, dy):
	if isnorthpolar(hp):
		if (dx ==  1) and (dy ==  0):
			return (hp + 1) % 4
		if (dx ==  0) and (dy ==  1):
			return (hp + 3) % 4
		if (dx ==  1) and (dy ==  1):
			return (hp + 2) % 4
		if (dx == -1) and (dy ==  0):
			return (hp + 4)
		if (dx ==  0) and (dy == -1):
			return 4 + ((hp + 1) % 4)
		if (dx == -1) and (dy == -1):
			return hp + 8
		return -1
	elif issouthpolar(hp):
		if (dx ==  1) and (dy ==  0):
			return 4 + ((hp + 1) % 4)
		if (dx ==  0) and (dy ==  1):
			return hp - 4
		if (dx == -1) and (dy ==  0):
			return 8 + ((hp + 3) % 4)
		if (dx ==  0) and (dy == -1):
			return 8 + ((hp + 1) % 4)
		if (dx == -1) and (dy == -1):
			return 8 + ((hp + 2) % 4)
		if (dx ==  1) and (dy ==  1):
			return hp - 8
		return -1
	else:
		if (dx ==  1) and (dy ==  0):
			return hp - 4
		if (dx ==  0) and (dy ==  1):
			return (hp + 3) % 4
		if (dx == -1) and (dy ==  0):
			return 8 + ((hp + 3) % 4)
		if (dx ==  0) and (dy == -1):
			return hp + 4
		if (dx ==  1) and (dy == -1):
			return 4 + ((hp + 1) % 4)
		if (dx == -1) and (dy ==  1):
			return 4 + ((hp - 1) % 4)
		return -1
	return -1

def get_neighbours(hp, nside):
	neighbours = []
	(base, x, y) = decompose_xy(hp, nside)

	# ( + , 0 )
	nx = (x + 1) % nside
	ny = y
	if x == (nside - 1):
		nbase = get_base_neighbour(base, 1, 0)
		if (isnorthpolar(base)):
			nx = x;
			# swap nx,ny
			(nx,ny) = (ny,nx)
	else:
		nbase = base

	neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( + , + )
	nx = (x + 1) % nside
	ny = (y + 1) % nside
	if (x == nside - 1) and (y == nside - 1):
		if (ispolar(base)):
			nbase = get_base_neighbour(base, 1, 1)
		else:
			nbase = -1
	elif (x == (nside - 1)):
		nbase = get_base_neighbour(base, 1, 0)
	elif (y == (nside - 1)):
		nbase = get_base_neighbour(base, 0, 1)
	else:
		nbase = base;

	if isnorthpolar(base):
		if (x == (nside - 1)):
			nx = nside - 1
		if (y == (nside - 1)):
			ny = nside - 1
		if (x == (nside - 1)) or (y == (nside - 1)):
			# swap nx,ny
			(nx,ny) = (ny,nx)

	if (nbase != -1):
		neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( 0 , + )
	nx = x
	ny = (y + 1) % nside
	if (y == (nside - 1)):
		nbase = get_base_neighbour(base, 0, 1)
		if (isnorthpolar(base)):
			ny = y
			# swap nx,ny
			(nx,ny) = (ny,nx)
	else:
		nbase = base

	neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( - , + )
	nx = (x + nside - 1) % nside
	ny = (y + 1) % nside
	if (x == 0) and (y == (nside - 1)):
		if isequatorial(base):
			nbase = get_base_neighbour(base, -1, 1)
		else:
			nbase = -1
	elif (x == 0):
		nbase = get_base_neighbour(base, -1, 0)
		if issouthpolar(base):
			nx = 0
			# swap nx,ny
			(nx,ny) = (ny,nx)
	elif (y == (nside - 1)):
		nbase = get_base_neighbour(base, 0, 1)
		if isnorthpolar(base):
			ny = y
			# swap nx,ny
			(nx,ny) = (ny,nx)
	else:
		nbase = base

	if (nbase != -1):
		neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( - , 0 )
	nx = (x + nside - 1) % nside
	ny = y
	if (x == 0):
		nbase = get_base_neighbour(base, -1, 0)
		if issouthpolar(base):
			nx = 0
			# swap nx,ny
			(nx,ny) = (ny,nx)
	else:
		nbase = base

	neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( - , - )
	nx = (x + nside - 1) % nside
	ny = (y + nside - 1) % nside
	if (x == 0) and (y == 0):
		if ispolar(base):
			nbase = get_base_neighbour(base, -1, -1)
		else:
			nbase = -1
	elif (x == 0):
		nbase = get_base_neighbour(base, -1, 0)
	elif (y == 0):
		nbase = get_base_neighbour(base, 0, -1)
	else:
		nbase = base;

	if issouthpolar(base):
		if (x == 0):
			nx = 0
		if (y == 0):
			ny = 0
		if (x == 0) or (y == 0):
			# swap nx,ny
			(nx,ny) = (ny,nx)

	if (nbase != -1):
		neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( 0 , - )
	ny = (y + nside - 1) % nside
	nx = x
	if (y == 0):
		nbase = get_base_neighbour(base, 0, -1)
		if issouthpolar(base):
			ny = y
			# swap nx,ny
			(nx,ny) = (ny,nx)
	else:
		nbase = base

	neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	# ( + , - )
	nx = (x + 1) % nside
	ny = (y + nside - 1) % nside
	if (x == (nside - 1)) and (y == 0):
		if isequatorial(base):
			nbase = get_base_neighbour(base, 1, -1)
		else:
			nbase = -1

	elif (x == (nside - 1)):
		nbase = get_base_neighbour(base, 1, 0)
		if isnorthpolar(base):
			nx = x
			# swap nx,ny
			(nx,ny) = (ny,nx)
	elif (y == 0):
		nbase = get_base_neighbour(base, 0, -1)
		if issouthpolar(base):
			ny = y
			# swap nx,ny
			(nx,ny) = (ny,nx)
	else:
		nbase = base

	if (nbase != -1):
		neighbours.append(nbase * (nside**2) + compose_xy(nx, ny, nside))

	return neighbours

def ispolar(hp):
	# the north polar healpixes are 0,1,2,3
	# the south polar healpixes are 8,9,10,11
	return (hp <= 3) or (hp >= 8)

def isequatorial(hp):
	# the north polar healpixes are 0,1,2,3
	# the south polar healpixes are 8,9,10,11
	return (hp >= 4) and (hp <= 7)

def isnorthpolar(hp):
	return (hp <= 3)

def issouthpolar(hp):
	return (hp >= 8)


# radius of the field, in degrees.
def get_closest_pow2_nside(radius):
	# 4 pi steradians on the sky, into 12 base-level healpixes
	area = 4. * pi / 12.
	# in radians:
	baselen = sqrt(area)
	n = baselen / (radians(radius*2.))
	p = max(0, int(round(log(n, 2.0))))
	return int(2**p)

def compose_xy(x, y, nside):
	return x*nside + y

# Returns (ralo,rahi, declo,dechi) in deg.
def healpix_radec_bounds(hp, nside):
	ralo = c_double()
	rahi = c_double()
	declo = c_double()
	dechi = c_double()
	_lib.healpix_radec_bounds(c_int(hp), c_int(nside),
							  byref(ralo), byref(rahi), byref(declo), byref(dechi))
	return (float(ralo.value), float(rahi.value), float(declo.value), float(dechi.value))

def xyztohealpix(x, y, z, nside):
	cx = ctypes.c_double(float(x))
	cy = ctypes.c_double(float(y))
	cz = ctypes.c_double(float(z))
	cns = ctypes.c_int(int(nside))
	chp = _lib.xyztohealpix(cx, cy, cz, cns)
	return chp

# Returns (bighp, hpx, hpy), bighp is int, hpx,hpy are floats.
def xyztohealpixf(x, y, z, nside):
	cx = ctypes.c_double(float(x))
	cy = ctypes.c_double(float(y))
	cz = ctypes.c_double(float(z))
	cns = ctypes.c_int(int(nside))
	cdx = ctypes.c_double(0)
	cdy = ctypes.c_double(0)
	chp = _lib.xyztohealpixf(cx, cy, cz, cns, byref(cdx), byref(cdy))
	(bighp, hpx, hpy) = decompose_xy(chp, nside)
	return (bighp, hpx + float(cdx.value), hpy + float(cdy.value))

# ra, dec in degrees
def radectohealpix(ra, dec, nside):
	(x,y,z) = radectoxyz(ra, dec)
	return xyztohealpix(x,y,z, nside)

# RA,Dec in degrees
# Returns (bighp, hpx, hpy), bighp is int, hpx,hpy are floats.
def radectohealpixf(ra, dec, nside):
	(x,y,z) = radectoxyz(ra, dec)
	return xyztohealpixf(x,y,z, nside)

# returns (ra,dec) in degrees.
def healpix_to_radec(hp, nside, dx=0.5, dy=0.5):
	cfunc = _lib.healpix_to_radecdeg
	ra	= c_double(0.)
	dec = c_double(0.)
	cfunc.argtypes = [c_int, c_int, c_double, c_double, c_void_p, c_void_p]
	cfunc(hp, nside, dx, dy, byref(ra), byref(dec))
	return (float(ra.value), float(dec.value))

def healpix_to_xyz(hp, nside, dx=0.5, dy=0.5):
	cfunc = _lib.healpix_to_xyz
	x = c_double(0.)
	y = c_double(0.)
	z = c_double(0.)
	cfunc.argtypes = [c_int, c_int, c_double, c_double, c_void_p, c_void_p, c_void_p]
	cfunc(hp, nside, dx, dy, byref(x), byref(y), byref(z))
	return (float(x.value), float(y.value), float(z.value))

def healpix_neighbours_within_range(ra, dec, radius, nside):
	cfunc = _lib.healpix_get_neighbours_within_range_radec
	cfunc.argtypes = [c_double, c_double, c_double, c_void_p, c_int]
	hps = (c_int * 9)(*tuple([-1]*9))
	nhps = cfunc(c_double(ra), c_double(dec), c_double(radius),
				 hps,
				 c_int(nside))
	pyhps = []
	for i in range(nhps):
		pyhps.append(int(hps[i]))
	return pyhps

# Returns True if the given healpix *may* overlap the given RA,Dec range.
# RAs should be in 0,360
def healpix_may_overlap_radec_range(hp, nside, ralo, rahi, declo, dechi):
	# check the four corners
	rds = [healpix_to_radec(hp, nside, dx, dy)
		   for (dx,dy) in [(0,0),(0,1),(1,1),(1,0)]]
	hpralo = min([r for (r,d) in rds])
	hprahi = max([r for (r,d) in rds])
	hpdeclo = min([d for (r,d) in rds])
	hpdechi = max([d for (r,d) in rds])
	return ((hpralo <= rahi) and (hprahi >= ralo) and
			(hpdeclo <= dechi) and (hpdechi >= declo))

class testhealpix(unittest.TestCase):
	def check_neighbours(self, hp, nside, truen):
		neigh = get_neighbours(hp, nside)
		print 'True(%3i): [ %s ]' % (hp, ', '.join([str(n) for n in truen]))
		print 'Got (%3i): [ %s ]' % (hp, ', '.join([str(n) for n in neigh]))
		self.assertEqual(len(neigh), len(truen))
		for n in truen:
			self.assert_(n in neigh)

	def test_within_range(self):
		print 'hps within 5 deg of (0,0):'
		hps = healpix_neighbours_within_range(0, 0, 5, 1)
		print hps
		self.assertEqual(hps, [4])

		print 'hps within 5 deg of (45,0):'
		hps = healpix_neighbours_within_range(45, 0, 5, 1)
		print hps
		self.assertEqual(set(hps), set([0,8,4,5]))

	def test_neighbours(self):
		# from test_healpix.c's output.
		self.check_neighbours(0, 4, [ 4, 5, 1, 77, 76, 143, 83, 87, ])
		self.check_neighbours(12, 4, [ 19, 23, 13, 9, 8, 91, 95, ])
		self.check_neighbours(14, 4, [ 27, 31, 15, 11, 10, 9, 13, 23, ])
		self.check_neighbours(15, 4, [ 31, 47, 63, 62, 11, 10, 14, 27, ])
		self.check_neighbours(27, 4, [ 31, 15, 14, 13, 23, 22, 26, 30, ])
		self.check_neighbours(108, 4, [ 32, 33, 109, 105, 104, 171, 175, 115, ])
		self.check_neighbours(127, 4, [ 51, 44, 40, 123, 122, 126, 50, ])
		self.check_neighbours(64, 4, [ 68, 69, 65, 189, 188, 131, 135, ])
		self.check_neighbours(140, 4, [ 80, 81, 141, 137, 136, 146, 147, ])
		self.check_neighbours(152, 4, [ 156, 157, 153, 149, 148, 161, 162, 163, ])
		self.check_neighbours(160, 4, [ 164, 165, 161, 148, 144, 128, 176, 177, ])
		self.check_neighbours(18, 4, [ 22, 23, 19, 95, 94, 93, 17, 21, ])
		self.check_neighbours(35, 4, [ 39, 29, 28, 111, 110, 34, 38, ])
		self.check_neighbours(55, 4, [ 59, 46, 45, 44, 51, 50, 54, 58, ])
		self.check_neighbours(191, 4, [ 67, 48, 124, 120, 187, 186, 190, 66, ])
		self.check_neighbours(187, 4, [ 191, 124, 120, 116, 183, 182, 186, 190, ])
		self.check_neighbours(179, 4, [ 183, 116, 112, 172, 168, 178, 182, ])
		self.check_neighbours(178, 4, [ 182, 183, 179, 172, 168, 164, 177, 181, ])


if __name__ == '__main__':
	unittest.main()
