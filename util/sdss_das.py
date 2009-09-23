from astrometry.util.run_command import run_command

def sdss_das_get_rcf(run, camcol, field, band, outfn=None, reruns=[40,41,42]):
	gotit = False
	for rerun in reruns:
		url = ('http://das.sdss.org/imaging/%i/%i/corr/%i/fpC-%06i-%s%i-%04i.fit.gz' %
			   (run, rerun, camcol, run, band, camcol, field))
		if outfn:
			outfn = outfn % { 'run':run, 'camcol':camcol, 'field':field, 'band':band }
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
