import os.path

def sdss_band_name(b):
	if b in ['u','g','r','i','z']:
		return b
	if b in [0,1,2,3,4]:
		return 'ugriz'[b]
	raise Exception('Invalid SDSS band: "' + str(b) + '"')

def sdss_band_index(b):
	if b in ['u','g','r','i','z']:
		return 'ugriz'.index(b)
	if b in [0,1,2,3,4]:
		return b
	raise Exception('Invalid SDSS band: "' + str(b) + '"')

def sdss_filename(filetype, run, camcol, field, band=None, rerun=0):
	if band is not None:
		band = sdss_band_name(band)
	x = dict(run=run, band=band, camcol=camcol, field=field, rerun=rerun)
	ftmap = {
		'fpC': 'fpC-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
		'fpAtlas': 'fpAtlas-%(run)06i-%(camcol)i-%(field)04i.fit',
		'fpM': 'fpM-%(run)06i-%(band)s%(camcol)i-%(field)04i.fit',
		'fpObjc': 'fpObjc-%(run)06i-%(camcol)i-%(field)04i.fit',
		'psField': 'psField-%(run)06i-%(camcol)i-%(field)04i.fit',
		'tsObj': 'tsObj-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',
		'tsField': 'tsField-%(run)06i-%(camcol)i-%(rerun)i-%(field)04i.fit',

		#'http://das.sdss.org/imaging/125/40/astrom/asTrans-000125.fit'
		# http://das.sdss.org/imaging/125/40/calibChunks/3/tsField-000125-3-40-0196.fit
		}
	format = ftmap.get(filetype, None)
	if format is None:
		return None
	#print 'format', format, 'x', x
	return format % x

def sdss_path(filetype, run, camcol, field, band=None, rerun=None):
	x = dict(run=run, band=band, camcol=camcol, field=field, rerun=rerun)
	y = (run, camcol, field, band, rerun)
	if filetype in ['fpC']:
		return '%(run)i/%(rerun)i/corr/%(camcol)i/' % x + sdss_filename(filetype, *y)
	elif filetype in ['psField', 'fpAtlas', 'fpObjc', 'fpM']:
		return '%(run)i/%(rerun)i/objcs/%(camcol)i/' % x + sdss_filename(filetype, *y)
	elif filetype in ['tsObj', 'tsField']:
		return '%(run)i/%(rerun)i/calibChunks/%(camcol)i/' % x + sdss_filename(filetype, *y)
	else:
		return None

def sdss_find_file(filetype, run, camcol, field, band=None, reruns=None, datadir=None, reduxdir=None):
	if filetype == 'psField':
		basedir = datadir
		for rerun in reruns:
			pth = os.path.join(basedir, sdss_path(filetype, run, camcol, field, band=band, rerun=rerun))
			print 'trying path', pth
			if os.path.exists(pth):
				return pth
	return None

