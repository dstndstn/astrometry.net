import os
import pyfits
from astrometry.util.pyfits_utils import fits_table
import numpy as np

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
	def getAsTrans(self):
		return self.astrans
	def getCalibVec(self):
		return self.calib

class PhotoObj(SdssFile):
	def __init__(self, *args, **kwargs):
		super(PhotoObj, self).__init__(*args, **kwargs)
		self.filetype = 'photoObj'
		self.table = None
	def getTable(self):
		return self.table

class runlist(object):
	pass

class DR8(DR7):
	_lup_to_mag_b = np.array([1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10])
	_two_lup_to_mag_b = 2.*_lup_to_mag_b
	_ln_lup_to_mag_b = np.log(_lup_to_mag_b)

	'''
	From
	http://data.sdss3.org/datamodel/glossary.html#asinh

	m = -(2.5/ln(10))*[asinh(f/2b)+ln(b)].

	The parameter b is a softening parameter measured in maggies, and
	for the [u, g, r, i, z] bands has the values
	[1.4, 0.9, 1.2, 1.8, 7.4] x 1e-10
	'''
	@staticmethod
	def luptitude_to_mag(Lmag, bandnum, badmag=25):
		if bandnum is None:
			# assume Lmag is broadcastable to a 5-vector
			twobi = DR8._two_lup_to_mag_b
			lnbi = DR8._ln_lup_to_mag_b
		else:
			twobi = DR8._two_lup_to_mag_b[bandnum]
			lnbi = DR8._ln_lup_to_mag_b[bandnum]
		# MAGIC -1.08.... = -2.5/np.log(10.)
		f = np.sinh(Lmag/-1.0857362047581294 - lnbi) * twobi
		# prevent log10(-flux)
		mag = np.zeros_like(f) + badmag
		I = (f > 0)
		mag[I] = -2.5 * np.log10(f[I])
		return mag

	@staticmethod
	def nmgy_to_mag(nmgy):
		return 22.5 - 2.5 * np.log10(nmgy)

	def __init__(self, **kwargs):
		'''
		Useful kwargs:
		
		basedir : (string) - local directory where data will be stored.
		'''
		DR7.__init__(self, **kwargs)
		# Local filenames
		self.filenames.update({
			'frame': 'frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits',
			'photoObj': 'photoObj-%(run)06i-%(camcol)i-%(field)04i.fits',
			})

		# URLs on DAS server
		self.dasurl = 'http://data.sdss3.org/sas/dr8/groups/boss/'
		self.daspaths = {
			'fpObjc': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
			'frame': 'photoObj/frames/%(rerun)s/%(run)i/%(camcol)i/frame-%(band)s-%(run)06i-%(camcol)i-%(field)04i.fits.bz2',
			'photoObj': 'photoObj/%(rerun)s/%(run)i/%(camcol)i/photoObj-%(run)06i-%(camcol)i-%(field)04i.fits',
			'psField': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/psField-%(run)06i-%(camcol)i-%(field)04i.fit',
			'fpM': 'photo/redux/%(rerun)s/%(run)i/objcs/%(camcol)i/fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
			}

		self.dassuffix = {
			'frame': '.bz2',
			'fpM': '.gz',
			}

		self.processcmds = {
			'frame': 'bunzip2 -cd %(input)s > %(output)s',
			'fpM': 'gunzip -cd %(input)s > %(output)s',
			}

		y = read_yanny(self._get_runlist_filename())
		y = y['RUNDATA']
		rl = runlist()
		rl.run = np.array(y['run'])
		rl.startfield = np.array(y['startfield'])
		rl.endfield = np.array(y['endfield'])
		rl.rerun = np.array(y['rerun'])
		#print 'Rerun type:', type(rl.rerun), rl.rerun.dtype
		self.runlist = rl

	def _get_runlist_filename(self):
		return self._get_data_file('runList-dr8.par')

	# read a data file describing the DR8 data
	def _get_data_file(self, fn):
		return os.path.join(os.path.dirname(__file__), fn)

	def get_rerun(self, run, field=None):
		I = (self.runlist.run == run)
		if field is not None:
			I *= (self.runlist.startfield <= field) * (self.runlist.endfield >= field)
		I = np.flatnonzero(I)
		reruns = np.unique(self.runlist.rerun[I])
		#print 'Reruns:', reruns
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
		#print 'URL:', url
		if self.curl:
			cmd = "curl -o '%(outfn)s' '%(url)s"
		else:
			cmd = "wget --continue -nv -O %(outfn)s '%(url)s'"

		# suffix to add to the downloaded filename
		suff = self.dassuffix.get(filetype, '')
		
		cmd = cmd % dict(outfn=outfn + suff, url=url)
		#print 'cmd:', cmd
		(rtn,out,err) = run_command(cmd)
		if rtn:
			print 'Command failed: command', cmd
			print 'Output:', out
			print 'Error:', err
			print 'Return val:', rtn
			return None

		if filetype in self.processcmds:
			cmd = self.processcmds[filetype]
			cmd = cmd % dict(input = outfn + suff, output = outfn)
			print 'cmd:', cmd
			(rtn,out,err) = run_command(cmd)
			if rtn:
				print 'Command failed: command', cmd
				print 'Output:', out
				print 'Error:', err
				print 'Return val:', rtn
				return None

		return outfn

	def readPhotoObj(self, run, camcol, field, filename=None):
		obj = PhotoObj(run, camcol, field)
		if filename is None:
			fn = self.getPath('photoObj', run, camcol, field)
		else:
			fn = filename
		obj.table = fits_table(fn)
		return obj

	def readFrame(self, run, camcol, field, band, filename=None):
		'''
		http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
		'''
		f = Frame(run, camcol, field, band)
		# ...
		if filename is None:
			fn = self.getPath('frame', run, camcol, field, band)
		else:
			fn = filename
		#print 'reading file', fn
 		p = pyfits.open(fn)
		#print 'got', len(p), 'HDUs'
		# in nanomaggies
		f.image = p[0].data
		# converts counts -> nanomaggies
		f.calib = p[1].data
		# table with val,x,y -- binned; use bilinear interpolation to expand
		sky = p[2].data
		f.sky = sky.field('allsky')[0]
		#print 'sky shape', f.sky.shape
		if len(f.sky.shape) != 2:
			f.sky = f.sky.reshape((-1, 256))
		f.skyxi = sky.field('xinterp')[0]
		f.skyyi = sky.field('yinterp')[0]
		#print 'p3:', p[3]
		# table -- asTrans structure
		tab = fits_table(p[3].data)
		assert(len(tab) == 1)
		tab = tab[0]
		# DR7 has NODE, INCL in radians...
		f.astrans = AsTrans(run, camcol, field, band,
							node=np.deg2rad(tab.node), incl=np.deg2rad(tab.incl),
							astrans=tab, cut_to_band=False)
							
		return f
	
