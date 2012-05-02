import os
import pyfits

from common import *
from dr7 import *
from astrometry.util.yanny import *
from astrometry.util.run_command import run_command

class Frame(SdssFile):
	def __init__(self, *args, **kwargs):
		super(Frame, self).__init__(*args, **kwargs)
		self.filetype = 'frame'
		self.image = None
	#def __str__(self):
	def getImage(self):
		return self.image

class runlist(object):
	pass

class DR8(DR7):
	def __init__(self, **kwargs):
		DR7.__init__(self, **kwargs)
		self.filenames.update({
			'frame': 'frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits',
			'photoObj': 'photoObj-%(run)06i-%(camcol)i-%(field)04i.fits',
			})

		self.daspaths = {
			'fpObjc': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
			'photoObj': 'photoObj/%(rerun)s/%(run)i/%(camcol)i/photoObj-%(run)06i-%(camcol)i-%(field)04i.fits',
			}

		y = read_yanny(self._get_data_file('runList-dr8.par'))
		y = y['RUNDATA']
		rl = runlist()
		rl.run = np.array(y['run'])
		rl.startfield = np.array(y['startfield'])
		rl.endfield = np.array(y['endfield'])
		rl.rerun = np.array(y['rerun'])
		print 'Rerun type:', type(rl.rerun), rl.rerun.dtype
		self.runlist = rl
		self.dasurl = 'http://data.sdss3.org/sas/dr8/groups/boss/'

	# read a data file describing the DR8 data
	def _get_data_file(self, fn):
		return os.path.join(os.path.dirname(__file__), fn)

	def get_rerun(self, run, field=None):
		I = (self.runlist.run == run)
		if field is not None:
			I *= (self.runlist.startfield <= field) * (self.runlist.endfield >= field)
		I = np.flatnonzero(I)
		reruns = np.unique(self.runlist.rerun[I])
		print 'Reruns:', reruns
		if len(reruns) == 0:
			return None
		return reruns[0]
	
	def retrieve(self, filetype, run, camcol, field, band=None, skipExisting=True):
		outfn = self.getPath(filetype, run, camcol, field, band)
		if outfn is None:
			return None
		if skipExisting and os.path.exists(outfn):
			return outfn
		rerun = self.get_rerun(run, field)
		path = self.daspaths[filetype]
		url = self.dasurl + path % dict(run=run, camcol=camcol, field=field, rerun=rerun,
										band=band)
		print 'URL:', url
		if self.curl:
			cmd = "curl -o '%(outfn)s' '%(url)s"
		else:
			cmd = "wget --continue -nv -O %(outfn)s '%(url)s'"
		cmd = cmd % dict(outfn=outfn, url=url)
		print 'cmd:', cmd
		(rtn,out,err) = run_command(cmd)
		if rtn:
			print 'Command failed: command', cmd
			print 'Output:', out
			print 'Error:', err
			print 'Return val:', rtn
			return None
		return outfn

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
	
