import os
import pyfits

from common import *

from astrometry.util.miscutils import *
from astrometry.util.pyfits_utils import *

class DR7(object):
	def __init__(self, curl=False):
		self.curl = curl
		# These are *LOCAL* filenames -- some are different than those
		# on the DAS.
		self.filenames = {
			'fpObjc': 'fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
			'fpM': 'fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
			'fpC': 'fpC-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
			'fpAtlas': 'fpAtlas-%(run)06i-%(camcol)i-%(field)04i.fit',
			'psField': 'psField-%(run)06i-%(camcol)i-%(field)04i.fit',
			#'tsObj': 'tsObj-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
			#'tsField': 'tsField-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
			'tsObj': 'tsObj-%(run)06i-%(camcol)i-%(field)04i.fit',
			'tsField': 'tsField-%(run)06i-%(camcol)i-%(field)04i.fit',
			}
		self.softbias = 1000
		self.basedir = None

	def getFilename(self, filetype, *args, **kwargs):
		for k,v in zip(['run', 'camcol', 'field', 'band'], args):
			kwargs[k] = v
		# convert band number to band character.
		if 'band' in kwargs:
			kwargs['band'] = band_name(kwargs['band'])
		if not filetype in self.filenames:
			return None
		pat = self.filenames[filetype]
		fn = pat % kwargs
		return fn

	def getPath(self, *args, **kwargs):
		fn = self.getFilename(*args, **kwargs)
		if self.basedir is not None:
			fn = os.path.join(self.basedir, fn)
		return fn

	def setBasedir(self, dirnm):
		self.basedir = dirnm

	def _open(self, fn):
		if self.basedir is not None:
			path = os.path.join(self.basedir, fn)
		else:
			path = fn
		return pyfits.open(path)

	def retrieve(self, filetype, run, camcol, field, band=None):
		# FIXME!
		from astrometry.util.sdss_das import sdss_das_get
		outfn = self.getFilename(filetype, run, camcol, field, band)
		sdss_das_get(filetype, outfn, run, camcol, field, band,
					 curl=self.curl)


	def readTsField(self, run, camcol, field, rerun):
		'''
		http://www.sdss.org/dr7/dm/flatFiles/tsField.html

		band: string ('u', 'g', 'r', 'i', 'z')
		'''
		f = TsField(run, camcol, field, rerun=rerun)
		fn = self.getFilename('tsField', run, camcol, field, rerun=rerun)
		print 'reading file', fn
		p = self._open(fn)
		print 'got', len(p), 'HDUs'
		f.setHdus(p)
		return f

	def readFpC(self, run, camcol, field, band):
		'''
		http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpC.html

		band: string ('u', 'g', 'r', 'i', 'z')
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
		for plane in [ 'INTER', 'SATUR', 'CR', 'GHOST' ]:
			M = fpM.getMaskPlane(plane)
			if M is None:
				continue
			for (c0,c1,r0,r1,coff,roff) in zip(M.cmin,M.cmax,M.rmin,M.rmax,
											   M.col0, M.row0):
				assert(coff == 0)
				assert(roff == 0)
				(outx,nil) = get_overlapping_region(c0-x0, c1+1-x0, 0, x1-x0)
				(outy,nil) = get_overlapping_region(r0-y0, r1+1-y0, 0, y1-y0)
				#print 'Mask col [%i, %i], row [%i, %i]' % (c0, c1, r0, r1)
				#print '  outx', outx, 'outy', outy
				ivarimg[outy,outx] = 0
		return ivarimg


