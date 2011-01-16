from astrometry.util.pyfits_utils import fits_table

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
	def __init__(self, run=None, camcol=None, field=None, band=None, **kwargs):
		self.run = run
		self.camcol = camcol
		self.field = field
		if band is not None:
			self.band = band
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


