import os
import os.path

def find_data_file(fn):
	datadir = os.environ.get('ASTROMETRY_DATA')
	if datadir:
		pth = os.path.join(datadir, fn)
		if os.path.exists(pth):
			return pth
	# Add smartness here...
	pth = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', fn)
	#print 'path', pth
	if os.path.exists(pth):
		return pth
	return None

