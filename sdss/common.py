from astrometry.util.pyfits_utils import fits_table

import numpy as np

def band_name(b):
	if b in ['u','g','r','i','z']:
		return b
	if b in [0,1,2,3,4]:
		return 'ugriz'[b]
	raise Exception('Invalid SDSS band: "' + str(b) + '"')

def band_index(b):
	if b in ['u','g','r','i','z']:
		return 'ugriz'.index(b)
	if b in [0,1,2,3,4]:
		return b
	raise Exception('Invalid SDSS band: "' + str(b) + '"')

class SdssFile(object):
	def __init__(self, run=None, camcol=None, field=None, band=None, rerun=None,
				 **kwargs):
		'''
		band: string ('u', 'g', 'r', 'i', 'z')
		'''
		self.run = run
		self.camcol = camcol
		self.field = field
		if band is not None:
			self.band = band_name(band)
			self.bandi = band_index(band)
		if rerun is not None:
			self.rerun = rerun
		self.filetype = 'unknown'

	def getRun(self):
		return self.__dict__.get('run', 0)
	def getCamcol(self):
		return self.__dict__.get('camcol', 0)
	def getField(self):
		return self.__dict__.get('field', 0)

	def __str__(self):
		s = 'SDSS ' + self.filetype
		s += ' %i-%i-%i' % (self.getRun(), self.getCamcol(), self.getField())
		if hasattr(self, 'band'):
			s += '-%s' % self.band
		return s


class AsTrans(SdssFile):
	'''
	In DR7, asTrans structures can appear in asTrans files (for a
	whole run) or in tsField files (in astrom/ or fastrom/).

	In DR8, they are in asTrans files, or in the "frames".
	'''
	def __init__(self, *args, **kwargs):
		'''
		node, incl: in radians
		'''
		# node=None, incl=None, astrans=None, 
		super(AsTrans, self).__init__(*args, **kwargs)
		self.filetype = 'asTrans'
		self.node = kwargs.get('node', None)
		self.incl = kwargs.get('incl', None)
		astrans = kwargs.get('astrans', None)
		# "astrans" must be an object with fields:
		#  {a,b,c,d,e,f}[band]
		#  {ricut}[band]
		#  {drow0, drow1, drow2, drow3, dcol0, dcol1, dcol2, dcol3}[band]
		#  {csrow, cscol, ccrow, cccol}[band]
		#self.astrans = astrans
		self.trans = {}
		if astrans is not None and hasattr(self, 'bandi'):
			for f in ['a','b','c','d','e','f', 'ricut',
					  'drow0', 'drow1', 'drow2', 'drow3',
					  'dcol0', 'dcol1', 'dcol2', 'dcol3',
					  'csrow', 'cscol', 'ccrow', 'cccol']:
				try:
					if hasattr(astrans, f):
						self.trans[f] = getattr(astrans, f)[self.bandi]
				except:
					pass

	def _get_abcdef(self):
		return tuple(self.trans[x] for x in 'abcdef')

	def _get_drow(self):
		return tuple(self.trans[x] for x in ['drow0', 'drow1', 'drow2', 'drow3'])

	def _get_dcol(self):
		return tuple(self.trans[x] for x in ['dcol0', 'dcol1', 'dcol2', 'dcol3'])

	def _get_cscc(self):
		return tuple(self.trans[x] for x in ['csrow', 'cscol', 'ccrow', 'cccol'])

	def _get_ricut(self):
		return self.trans['ricut']

	def pixel_to_radec(self, x, y, color):
		mu, nu = self.pixel_to_munu(x, y, color)
		return self.munu_to_radec(mu, nu)

	def radec_to_pixel(self, ra, dec, color):
		mu, nu = self.radec_to_munu(ra, dec)
		return self.munu_to_pixel(mu, nu, color)
	
	def munu_to_pixel(self, mu, nu, color):
		a, b, c, d, e, f = self._get_abcdef()
		determinant = b * f - c * e
		B = f  / determinant
		C = -c / determinant
		E = -e / determinant
		F = b  / determinant
		yprime = B * (mu - a) + C * (nu - d)
		xprime = E * (mu - a) + F * (nu - d)
		return self.prime_to_pixel(xprime, yprime, color)

	def pixel_to_munu(self, x, y, color):
		(xprime, yprime) = self.pixel_to_prime(x, y, color)
		a, b, c, d, e, f = self._get_abcdef()
		mu = a + b * yprime + c * xprime
		nu = d + e * yprime + f * xprime
		return (mu, nu)

	def pixel_to_prime(self, x, y, color):
		# Secret decoder ring:
		#  http://www.sdss.org/dr7/products/general/astrometry.html
		# (color)0 is called riCut;
		# g0, g1, g2, and g3 are called
		#    dRow0, dRow1, dRow2, and dRow3, respectively;
		# h0, h1, h2, and h3 are called
		#    dCol0, dCol1, dCol2, and dCol3, respectively;
		# px and py are called csRow and csCol, respectively;
		# and qx and qy are called ccRow and ccCol, respectively.
		color0 = self._get_ricut()
		g0, g1, g2, g3 = self._get_drow()
		h0, h1, h2, h3 = self._get_dcol()
		px, py, qx, qy = self._get_cscc()

		# #$(%*&^(%$%*& bad documentation.
		(px,py) = (py,px)
		(qx,qy) = (qy,qx)

		yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
		xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3

		# The code below implements this, vectorized:
		# if color < color0:
		#	xprime += px * color
		#	yprime += py * color
		# else:
		#	xprime += qx
		#	yprime += qy
		color  = np.atleast_1d(color)
		color0 = np.atleast_1d(color0)
		qx = qx * np.ones_like(x)
		qy = qy * np.ones_like(y)
		#print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
		xprime += np.where(color < color0, px * color, qx)
		yprime += np.where(color < color0, py * color, qy)
		return (xprime, yprime)

	def prime_to_pixel(self, xprime, yprime,  color):
		color0 = self._get_ricut()
		g0, g1, g2, g3 = self._get_drow()
		h0, h1, h2, h3 = self._get_dcol()
		px, py, qx, qy = self._get_cscc()

		# #$(%*&^(%$%*& bad documentation.
		(px,py) = (py,px)
		(qx,qy) = (qy,qx)

		color  = np.atleast_1d(color)
		color0 = np.atleast_1d(color0)
		# FIXME -- get the broadcasting right...
		qx = qx * np.ones_like(xprime)
		qy = qy * np.ones_like(yprime)
		#print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
		xprime -= np.where(color < color0, px * color, qx)
		yprime -= np.where(color < color0, py * color, qy)

		# Now invert:
		#   yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
		#   xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
		x = xprime - h0
		# dumb-ass Newton's method
		dx = 1.
		# FIXME -- should just update the ones that aren't zero
		# FIXME -- should put in some failsafe...
		while max(np.abs(np.atleast_1d(dx))) > 1e-10:
			xp    = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
			dxpdx = 1 +      h1     + h2 * 2*x +  h3 * 3*x**2
			dx = (xprime - xp) / dxpdx
			#print 'Max Newton dx', max(abs(dx))
			x += dx
		y = yprime - (g0 + g1 * x + g2 * x**2 + g3 * x**3)
		return (x, y)

	def radec_to_munu(self, ra, dec):
		node,incl = self.node, self.incl
		ra, dec = np.deg2rad(ra), np.deg2rad(dec)
		mu = node + np.arctan2(np.sin(ra - node) * np.cos(dec) * np.cos(incl) +
							   np.sin(dec) * np.sin(incl),
							   np.cos(ra - node) * np.cos(dec))
		nu = np.arcsin(-sin(ra - node) * np.cos(dec) * np.sin(incl) +
					   np.sin(dec) * np.cos(incl))
		mu, nu = np.rad2deg(mu), np.rad2deg(nu)
		mu += (360. * (mu < 0))
		return (mu, nu)

	def munu_to_radec(self, mu, nu):
		node,incl = self.node, self.incl
		mu, nu = np.deg2rad(mu), np.deg2rad(nu)
		ra = node + np.arctan2(np.sin(mu - node) * np.cos(nu) * np.cos(incl) -
							   np.sin(nu) * np.sin(incl),
							   np.cos(mu - node) * np.cos(nu))
		dec = np.arcsin(np.sin(mu - node) * np.cos(nu) * np.sin(incl) +
						np.sin(nu) * np.cos(incl))
		ra, dec = np.rad2deg(ra), np.rad2deg(dec)
		ra += (360. * (ra < 0))
		return (ra, dec)


class TsField(SdssFile):
	def __init__(self, *args, **kwargs):
		super(TsField, self).__init__(*args, **kwargs)
		self.filetype = 'tsField'
	def setHdus(self, p):
		self.hdus = p
	def getAsTrans(self, band):
		bandi = band_index(band)
		band = band_name(band)
		T = fits_table(self.hdus[1].data)
		T = T[0]
		#node,incl = self.getNode(), self.getIncl()
		hdr = self.hdus[0].header
		node = np.deg2rad(hdr.get('NODE'))
		incl = np.deg2rad(hdr.get('INCL'))
		asTrans = AsTrans(self.run, self.camcol, self.field, band=band,
						  node=node, incl=incl, astrans=T)
		return asTrans
	
class FpObjc(SdssFile):
	def __init__(self, *args, **kwargs):
		super(FpObjc, self).__init__(*args, **kwargs)
		self.filetype = 'fpObjc'

class FpM(SdssFile):
	def __init__(self, *args, **kwargs):
		super(FpM, self).__init__(*args, **kwargs)
		self.filetype = 'fpM'

	def setHdus(self, p):
		self.hdus = p

	def getMaskPlane(self, name):
		# HACK -- this must be described somewhere sensible...
		maskmap = { 'INTER': 0,
					'SATUR': 1,
					'CR'   : 8,
					'GHOST': 9,
					}
		if not name in maskmap:
			raise RuntimeException('Unknown mask plane \"%s\"' % name)

		return fits_table(self.hdus[1 + maskmap[name]].data)

class FpC(SdssFile):
	def __init__(self, *args, **kwargs):
		super(FpC, self).__init__(*args, **kwargs)
		self.filetype = 'fpC'
	def getImage(self):
		return self.image

class PsField(SdssFile):
	def __init__(self, *args, **kwargs):
		super(PsField, self).__init__(*args, **kwargs)
		self.filetype = 'psField'

	def setHdus(self, p):
		t = fits_table(p[6].data)
		# the table has only one row...
		assert(len(t) == 1)
		t = t[0]
		self.gain = t.gain
		self.dark_variance = t.dark_variance
		self.sky = t.sky
		self.skyerr = t.skyerr
		self.psp_status = t.status
		# Double-Gaussian PSF params
		self.dgpsf_s1 = t.psf_sigma1_2g
		self.dgpsf_s2 = t.psf_sigma2_2g
		self.dgpsf_b  = t.psf_b_2g

	def getDoubleGaussian(self, bandnum):
		# http://www.sdss.org/dr7/dm/flatFiles/psField.html
		# good = PSP_FIELD_OK
		status = self.psp_status[bandnum]
		if status != 0:
			print 'Warning: PsField status[band=%s] =' % (bandnum, status)
		a  = 1.0
		s1 = self.dgpsf_s1[bandnum]
		s2 = self.dgpsf_s2[bandnum]
		b  = self.dgpsf_b[bandnum]
		return (float(a), float(s1), float(b), float(s2))

	def getGain(self, band=None):
		if band is not None:
			return self.gain[band]
		return self.gain

	def getDarkVariance(self, band=None):
		if band is not None:
			return self.dark_variance[band]
		return self.dark_variance

	def getSky(self, band=None):
		if band is not None:
			return self.sky[band]
		return self.sky

	def getSkyErr(self, band=None):
		if band is not None:
			return self.skyerr[band]
		return self.skyerr


