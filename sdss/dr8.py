import pyfits

from common import *
from dr7 import *

class Frame(SdssFile):
	def __init__(self, *args, **kwargs):
		super(Frame, self).__init__(*args, **kwargs)
		self.filetype = 'frame'
		self.image = None
	#def __str__(self):
	def getImage(self):
		return self.image

class DR8(DR7):
	def __init__(self):
		self.filenames.update({
			'frame': 'frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits',
			})

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
	
