from lsst.afw.detection import SourceSet, Source
from astrometry.util.pyfits_utils import *

sourceset_fields = ['FlagForDetection', "XAstrom", "XAstromErr", "YAstrom", "YAstromErr",
					"PsfFlux", "ApFlux", "Ixx", "IxxErr", "Iyy",
					"IyyErr", "Ixy", "IxyErr"]

# eg, from astrometry.util.pyfits_utils : fits_table() or text_table()
def sourceset_from_table(t):
	N = len(t)
	ss = SourceSet()
	for i in range(N):
		s = Source()
		ss.push_back(s)
	for f in sourceset_fields:
		vals = t.getcolumn(f.lower())
		for s,v in zip(ss,vals):
			fname = "set" + f
			func = getattr(s, fname)
			if func is None:
				raise Exception('Function not found in Source object: ' + fname + ', object %s' % str(s))
			func(v)
	return ss
	

def sourceset_to_dict(ss):
	d = dict()
	for f in sourceset_fields:
		vals = []
		for s in ss:
			func = getattr(s, "get" + f)
			vals.append(func())
		d[f] = vals
	return d

def sourceset_to_table(ss):
	sd = sourceset_to_dict(ss)
	td = tabledata()
	for k,v in sd.items():
		td.set(k, array(v))
	return td

def sourceset_from_dict(d):
	x = d[sourceset_fields[0]]
	N = len(x)
	ss = SourceSet()
	for i in range(N):
		s = Source()
		ss.push_back(s)

	for f in sourceset_fields:
		vals = d[f]
		for s,v in zip(ss,vals):
			func = getattr(s, "set" + f)
			func(v)

	return ss
