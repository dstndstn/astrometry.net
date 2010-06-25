from lsst.afw.detection import SourceSet, Source

# eg, from astrometry.util.pyfits_utils : fits_table() or text_table()
def sourceset_from_table(t):
    N = len(t)
    ss = SourceSet()
    for i in range(N):
        s = Source()
        ss.push_back(s)
    for f in ["XAstrom", "XAstromErr", "YAstrom", "YAstromErr",
              "PsfFlux", "ApFlux", "Ixx", "IxxErr", "Iyy",
              "IyyErr", "Ixy", "IxyErr"]:
		f = f.lower()
        vals = t.getcolumn(f)
        for s,v in zip(ss,vals):
            func = getattr(s, "set" + f)
            func(v)
    return ss
	

def sourceset_to_dict(ss):
    d = dict()
    for f in ["XAstrom", "XAstromErr", "YAstrom", "YAstromErr",
              "PsfFlux", "ApFlux", "Ixx", "IxxErr", "Iyy",
              "IyyErr", "Ixy", "IxyErr"]:
        vals = []
        for s in ss:
            func = getattr(s, "get" + f)
            vals.append(func())
        d[f] = vals
    return d

def sourceset_from_dict(d):
    x = d['XAstrom']
    N = len(x)
    ss = SourceSet()
    for i in range(N):
        s = Source()
        ss.push_back(s)

    for f in ["XAstrom", "XAstromErr", "YAstrom", "YAstromErr",
              "PsfFlux", "ApFlux", "Ixx", "IxxErr", "Iyy",
              "IyyErr", "Ixy", "IxyErr"]:
        vals = d[f]
        for s,v in zip(ss,vals):
            func = getattr(s, "set" + f)
            func(v)

    return ss
