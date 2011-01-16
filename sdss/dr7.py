import pyfits

from common import *

class DR7(object):
	def __init__(self):
		self.filenames = {
			'fpObjc': 'fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
			'fpM': 'fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
			'psField': 'psField-%(run)06i-%(camcol)i-%(field)04i.fit',
			}

	def getFilename(self, filetype, *args, **kwargs):
		for k,v in zip(['run', 'camcol', 'field', 'band'], args):
			kwargs[k] = v
		if not filetype in self.filenames:
			return None
		pat = self.filenames[filetype]
		return pat % kwargs

	def _open(self, fn):
		return pyfits.open(fn)

	def readFrame(self, run, camcol, field, band, filename=None):
		'''
		http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
		'''
		f = Frame(run, camcol, field, band)
		# ...
		if filename is None:
			fn = self.getFilename('frame', run, camcol, field, band)
		else:
			fn = filename
		print 'reading file', fn
 		p = self._open(fn)
		print 'got', len(p), 'HDUs'
		# in nanomaggies?
		f.image = p[0].data
		# converts counts -> nanomaggies
		f.calib = p[1].data
		# table with val,x,y -- binned; use bilinear interpolation to expand
		f.sky = p[2].data
		#print 'p3:', p[3]
		# table -- asTrans structure
		f.astrans = p[3].data
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
		return f

