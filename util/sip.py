# This file is part of the Astrometry.net suite.
# Copyright 2006, 2007 Keir Mierle and Dustin Lang.
#
# The Astrometry.net suite is free software; you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, version 2.
#
# The Astrometry.net suite is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the Astrometry.net suite ; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA		 02110-1301 USA

import ctypes
from ctypes import *
import ctypes.util
import sys


# All this jazz was in here because I was having trouble getting
# the _sip.so library loaded - the library being in the same directory
# as the .py file doesn't help, nor does having it on your PYTHONPATH -
# you just have to stick it on your LD_LIBRARY_PATH, as far as I can
# see - unless you give the absolute path.
_sip = None
_libname = ctypes.util.find_library('_sip.so')
if _libname:
	#print 'libname is ', _libname
	_sip = ctypes.CDLL(_libname)
else:
	import os.path
	p = os.path.join(os.path.dirname(__file__), '_sip.so')
	if os.path.exists(p):
		_sip = ctypes.CDLL(p)
		#if _sip:
		#	 print 'loaded', p
	else:
		print 'file does not exist:', p

def loadlibrary(fn):
	global _sip
	#print 'loading library ', fn
	_sip = ctypes.CDLL(fn)

def libraryloaded():
	return _sip is not None

class Tan(ctypes.Structure):
	_fields_ = [("crval", c_double*2),
				("crpix", c_double*2),
				("cd",	  c_double*4),
				("imagew", c_double),
				("imageh", c_double)]

	def __init__(self, filename=None):
		if filename is not None:
			cfn = c_char_p(filename)
			rtn = _sip.tan_read_header_file(cfn, ctypes.pointer(self))
			if not rtn:
				raise Exception, 'Failed to parse TAN header from file "%s"' % filename

	def __str__(self):
		return ('<Tan: CRVAL (%f, %f)' % (self.crval[0], self.crval[1]) +
				' CRPIX (%f, %f)' % (self.crpix[0], self.crpix[1]) +
				' CD (%f, %f; %f %f)' % (self.cd[0], self.cd[1], self.cd[2], self.cd[3]) +
				' Image size (%f, %f)>' % (self.imagew, self.imageh)
				)

	# returns (ra,dec) in degrees.
	def pixelxy2radec(self, px,py):
		'Return ra,dec of px,py'
		ra = ctypes.c_double(3.14159)
		dec = ctypes.c_double(2.71828)
		fpx = ctypes.c_double(px)
		fpy = ctypes.c_double(py)
		_sip.tan_pixelxy2radec(
				ctypes.pointer(self),
				fpx, fpy,
				ctypes.pointer(ra),
				ctypes.pointer(dec))
		return ra.value, dec.value

	def radec2pixelxy(self, RA, Dec):
		'Return px,py of ra,dec'
		ra = ctypes.c_double(RA)
		dec = ctypes.c_double(Dec)
		fpx = ctypes.c_double(0.)
		fpy = ctypes.c_double(0.)
		_sip.tan_radec2pixelxy(
				ctypes.pointer(self),
				ra, dec,
				ctypes.pointer(fpx),
				ctypes.pointer(fpy)
				)
		return fpx.value, fpy.value

	def radec_bounds(self, stepsize):
		ramin = ctypes.c_double(0)
		ramax = ctypes.c_double(0)
		decmin = ctypes.c_double(0)
		decmax = ctypes.c_double(0)
		step = ctypes.c_int(stepsize)
		sip = Sip()
		sip.wcstan = self
		sip.a_order = 0
		sip.b_order = 0
		sip.ap_order = 0
		sip.bp_order = 0
		_sip.sip_get_radec_bounds(
			#ctypes.pointer(self),
			ctypes.pointer(sip),
			step,
			ctypes.pointer(ramin),
			ctypes.pointer(ramax),
			ctypes.pointer(decmin),
			ctypes.pointer(decmax)
			)
		return (ramin.value, ramax.value, decmin.value, decmax.value)

	def write_to_file(self, fn):
		if fn is None:
			raise Exception, "Can't have None filename."
		cfn = c_char_p(fn)
		rtn = _sip.tan_write_to_file(ctypes.pointer(self), cfn)
		return rtn

	# in arcsec/pixel
	def get_pixel_scale(self):
		_sip.tan_pixel_scale.restype = c_double
		return float(_sip.tan_pixel_scale(ctypes.pointer(self)))
	
	def __str__(self):
		return '<Tan: crval=(%g, %g), crpix=(%g, %g), cd=(%g, %g; %g, %g), imagew=%d, imageh=%d>' % \
			   (self.crval[0], self.crval[1], self.crpix[0], self.crpix[1], self.cd[0], self.cd[1],
				self.cd[2], self.cd[3], self.imagew, self.imageh)

SIP_MAXORDER = 10

class Sip(ctypes.Structure):
	_fields_ = [('wcstan', Tan),
				('a_order', c_int),
				('b_order', c_int),
				('a', c_double*(SIP_MAXORDER**2)),
				('b', c_double*(SIP_MAXORDER**2)),
				('ap_order', c_int),
				('bp_order', c_int),
				('ap', c_double*(SIP_MAXORDER**2)),
				('bp', c_double*(SIP_MAXORDER**2)),]

	def __init__(self, filename=None):
		if not filename is None:
			cfn = c_char_p(filename)
			rtn = _sip.sip_read_header_file(cfn, ctypes.pointer(self))
			if not rtn:
				raise Exception, 'Failed to parse SIP header from file "%s"' % filename

	def __str__(self):
		return '<Sip: ' + str(self.wcstan) + \
			   ', a_order=%d, b_order=%d, ap_order=%d>' % (self.a_order, self.b_order, self.ap_order)

	def get_a_term(self, i, j):
		return Sip.get_term(self.a, i, j)
	def get_b_term(self, i, j):
		return Sip.get_term(self.b, i, j)
	def get_ap_term(self, i, j):
		return Sip.get_term(self.ap, i, j)
	def get_bp_term(self, i, j):
		return Sip.get_term(self.bp, i, j)

	@staticmethod
	def get_term(arr, i, j):
		return arr[i * SIP_MAXORDER + j]

	def set_a_term(self, i, j, c):
		Sip.set_term(self.a, i, j, c)
	def set_b_term(self, i, j, c):
		Sip.set_term(self.b, i, j, c)
	def set_ap_term(self, i, j, c):
		Sip.set_term(self.ap, i, j, c)
	def set_bp_term(self, i, j, c):
		Sip.set_term(self.bp, i, j, c)

	@staticmethod
	def set_term(arr, i, j, c):
		arr[i * SIP_MAXORDER + j] = c

	def set_a_terms(self, terms):
		set_terms(self.a, terms)
	def set_b_terms(self, terms):
		set_terms(self.b, terms)
	def set_ap_terms(self, terms):
		set_terms(self.ap, terms)
	def set_bp_terms(self, terms):
		set_terms(self.bp, terms)

	@staticmethod
	def set_terms(arr, terms):
		for (i, j, c) in terms:
			set_term(arr, i, j, c)

	# returns a list of (i, j, coeff) tuples.
	def get_nonzero_a_terms(self):
		return Sip.nonzero_terms(self.a, self.a_order)
	def get_nonzero_b_terms(self):
		return Sip.nonzero_terms(self.b, self.b_order)
	def get_nonzero_ap_terms(self):
		return Sip.nonzero_terms(self.ap, self.ap_order)
	def get_nonzero_bp_terms(self):
		return Sip.nonzero_terms(self.bp, self.bp_order)

	@staticmethod
	def nonzero_terms(arr, order):
		terms = []
		for i in range(order):
			for j in range(order):
				if i+j > order:
					continue
				c = Sip.get_term(arr, i, j)
				if c != 0:
					terms.append((i, j, c))
		return terms

	def write_to_file(self, fn):
		if fn is None:
			raise Exception, "Can't have None filename."
		cfn = c_char_p(fn)
		rtn = _sip.sip_write_to_file(ctypes.pointer(self), cfn)
		return rtn


	def pixelxy2radec(self, px,py):
		'Return ra,dec of px,py'
		ra = ctypes.c_double(0)
		dec = ctypes.c_double(0)
		fpx = ctypes.c_double(px)
		fpy = ctypes.c_double(py)
		_sip.sip_pixelxy2radec(
				ctypes.pointer(self),
				fpx, fpy,
				ctypes.pointer(ra),
				ctypes.pointer(dec))
		return ra.value, dec.value

	def radec2pixelxy(self, RA, Dec):
		'Return px,py of ra,dec'
		ra = ctypes.c_double(RA)
		dec = ctypes.c_double(Dec)
		fpx = ctypes.c_double(0.)
		fpy = ctypes.c_double(0.)
		_sip.sip_radec2pixelxy(
				ctypes.pointer(self),
				ra, dec,
				ctypes.pointer(fpx),
				ctypes.pointer(fpy))
		return fpx.value, fpy.value

	def radec_bounds(self, stepsize=50):
		ramin = ctypes.c_double(0)
		ramax = ctypes.c_double(0)
		decmin = ctypes.c_double(0)
		decmax = ctypes.c_double(0)
		step = ctypes.c_int(stepsize)
		_sip.sip_get_radec_bounds(
			ctypes.pointer(self),
			step,
			ctypes.pointer(ramin),
			ctypes.pointer(ramax),
			ctypes.pointer(decmin),
			ctypes.pointer(decmax)
			)
		return (ramin.value, ramax.value, decmin.value, decmax.value)


if __name__ == '__main__':
	t= Tan()
	t.crval[0] = 0.0
	t.crval[1] = 0.0
	t.crpix[0] = 0.0
	t.crpix[1] = 0.0
	t.cd[0] = 1.0
	t.cd[1] = 0.0
	t.cd[2] = 0.0
	t.cd[3] = 1.0

	ra,dec = t.pixelxy2radec(2.0,3.0)
	print ra,dec

	s = Sip()
	s.wcstan = t
	s.a_order = 1
	s.b_order = 1
	s.ap_order = 1
	s.bp_order = 1

	ra,dec = s.pixelxy2radec(2.0,3.0)
	print ra,dec

	if len(sys.argv) > 1:
		s = Sip(sys.argv[1])
		print s
		ra,dec = s.pixelxy2radec(2.0,3.0)
		print ra,dec

