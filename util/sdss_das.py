from astrometry.util.run_command import run_command

def get_urls(urls, outfn):
	gotit = False
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


def sdss_das_get_fpc(run, camcol, field, band, outfn=None, reruns=[40,41,42,44]):
	urls = [('http://das.sdss.org/imaging/%i/%i/corr/%i/fpC-%06i-%s%i-%04i.fit.gz' %
		 (run, rerun, camcol, run, band, camcol, field))
		for rerun in reruns]
	if outfn:
		outfn = outfn % { 'run':run, 'camcol':camcol, 'field':field, 'band':band }
	return get_urls(urls, outfn)

def sdss_das_get_mask(run, camcol, field, band, outfn=None, reruns=[40,41,42,44]):
	urls = [('http://das.sdss.org/imaging/%i/%i/objcs/%i/fpM-%06i-%s%i-%04i.fit' %
		 (run, rerun, camcol, run, band, camcol, field))
		for rerun in reruns]
	if outfn:
		outfn = outfn % { 'run':run, 'camcol':camcol, 'field':field, 'band':band }
	return get_urls(urls, outfn)
