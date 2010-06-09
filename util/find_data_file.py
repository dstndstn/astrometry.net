import os
import os.path

def find_data_file(fn):
	searched = []
	datadir = os.environ.get('ASTROMETRY_DATA')
	if datadir:
		pth = os.path.join(datadir, fn)
		if os.path.exists(pth):
			return pth
		searched.append(pth)
	# Add smartness here...
	dirnm = os.path.dirname(__file__)
	for i in range(5):
		pth = os.path.join(dirnm, 'data', fn)
		if os.path.exists(pth):
			return pth
		searched.append(pth)
		dirnm = os.path.dirname(dirnm)
	#pth = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', fn)
	#print 'path', pth
	#if os.path.exists(pth):
	#		return pth
	print 'Failed to find data file:', fn
	print '  searched paths:', searched
	return None

