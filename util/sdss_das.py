from astrometry.util.run_command import run_command
from astrometry.util.sdss_filenames import *

def get_urls(urls, outfn):
	for url in urls:
		cmd = 'wget --continue -nv '
		if outfn:
			cmd += '-O %s ' % outfn
		cmd += '\"%s\"' % url
		print 'Running:', cmd
		(rtn, out, err) = run_command(cmd)
		if rtn == 0:
			return True
		if rtn:
			print 'Command failed: command', cmd
			print 'Output:', out
			print 'Error:', err
			print 'Return val:', rtn
	return False

def sdss_das_get(filetype, outfn, run, camcol, field, band=None, reruns=None, suffix=''):
	if reruns is None:
		reruns = [40,41,42,44]
	urls = ['http://das.sdss.org/imaging/' +
			sdss_path(filetype, run, camcol, field, band, rerun) +
			suffix
			for rerun in reruns]
	if outfn:
		outfn = outfn % { 'run':run, 'camcol':camcol, 'field':field, 'band':band }
	return get_urls(urls, outfn)

def sdss_das_get_fpc(run, camcol, field, band, outfn=None, reruns=None):
	return sdss_das_get('fpC', outfn, run, camcol, field, band, reruns, suffix='.gz')

def sdss_das_get_mask(run, camcol, field, band, outfn=None, reruns=None):
	return sdss_das_get('fpM', outfn, run, camcol, field, band, reruns)


