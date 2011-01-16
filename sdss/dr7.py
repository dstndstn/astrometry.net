import pyfits

from common import *

from astrometry.util.miscutils import *
from astrometry.util.pyfits_utils import *

class DR7(object):
	def __init__(self):
		self.filenames = {
			'fpObjc': 'fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
			'fpM': 'fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
			'fpC': 'fpC-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
			'fpAtlas': 'fpAtlas-%(run)06i-%(camcol)i-%(field)04i.fit',
			'psField': 'psField-%(run)06i-%(camcol)i-%(field)04i.fit',
			'tsObj': 'tsObj-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
			'tsField': 'tsField-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
			}
		self.softbias = 1000

	def getFilename(self, filetype, *args, **kwargs):
		for k,v in zip(['run', 'camcol', 'field', 'band'], args):
			kwargs[k] = v
		if not filetype in self.filenames:
			return None
		pat = self.filenames[filetype]
		return pat % kwargs

	def _open(self, fn):
		return pyfits.open(fn)

	def readFpC(self, run, camcol, field, band):
		'''
		http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpC.html
		'''
		f = FpC(run, camcol, field, band)
		# ...
		fn = self.getFilename('fpC', run, camcol, field, band)
		print 'reading file', fn
		p = self._open(fn)
		print 'got', len(p), 'HDUs'
		f.image = p[0].data
		return f

	def readFpObjc(self, run, camcol, field):
		'''
		http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpObjc.html
		'''
		f = FpObjc(run, camcol, field)
		# ...
		fn = self.getFilename('fpObjc', run, camcol, field)
		print 'reading file', fn
		p = self._open(fn)
		print 'got', len(p), 'HDUs'
		return f

	def readFpM(self, run, camcol, field, band):
		'''
		http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
		'''
		f = FpM(run, camcol, field, band)
		# ...
		fn = self.getFilename('fpM', run, camcol, field, band)
		print 'reading file', fn
		p = self._open(fn)
		print 'got', len(p), 'HDUs'
		f.setHdus(p)
		return f

	def readPsField(self, run, camcol, field):
		'''
		http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/psField.html
		'''
		f = PsField(run, camcol, field)
		# ...
		fn = self.getFilename('psField', run, camcol, field)
		print 'reading file', fn
		p = self._open(fn)
		print 'got', len(p), 'HDUs'
		f.setHdus(p)
		return f


	def getInvvar(self, fpC, fpM, gain, darkvar, sky, skyerr,
				  x0=0, x1=None, y0=0, y1=None):
		'''
		Produces a (cut-out of) the inverse-variance noise image, from columns
		[x0,x1] and rows [y0,y1] (inclusive).  Default is the whole image.

		fpC is the image pixels (eg FpC.getImage())
		fpM is the FpM
		gain, darkvar, sky, and skyerr can be retrieved from the psField file.
		'''
		if x1 is None:
			x1 = fpC.shape[1]-1
		if y1 is None:
			y1 = fpC.shape[0]-1

		# Poisson: mean = variance
		# Add readout noise?
		# Spatial smoothing?
		img = fpC[y0:y1+1, x0:x1+1]

		# from http://www.sdss.org/dr7/algorithms/fluxcal.html
		ivarimg = 1./((img + sky) / gain + darkvar + skyerr)

		# Noise model:
		#  -mask coordinates are wrt fpC coordinates.
		#  -INTERP, SATUR, CR,
		#  -GHOST?
		# HACK -- MAGIC -- these are the indices of INTER, SATUR, CR, and GHOST
		for plane in [ 'INTER', 'SATUR', 'CR', 'GHOST' ]:
			M = fpM.getMaskPlane(plane)
			if M is None:
				continue
			for (c0,c1,r0,r1) in zip(M.cmin,M.cmax,M.rmin,M.rmax):
				(outx,nil) = get_overlapping_region(c0-x0, c1+1-x0, 0, x1-x0)
				(outy,nil) = get_overlapping_region(r0-y0, r1+1-y0, 0, y1-y0)
				ivarimg[outy,outx] = 0
		return ivarimg


