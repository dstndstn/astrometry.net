from astrometry.util.file import *

class CallGlobal(object):
	def __init__(self, pattern, *args, **kwargs):
		self.pat = pattern
		self.args = args
		self.kwargs = kwargs
	def __call__(self, stage, kwargs):
		func = self.pat % stage
		kwa = self.kwargs.copy()
		kwa.update(kwargs)
		return func(*self.args, **kwa)

def runstage(stage, picklepat, stagefunc, force=[], prereqs={},
			 update=True, **kwargs):
	print 'Runstage', stage

	pfn = picklepat % stage
	if os.path.exists(pfn):
		if stage in force:
			print 'Ignoring pickle', pfn, 'and forcing stage', stage
		else:
			print 'Reading pickle', pfn
			R = unpickle_from_file(pfn)
			return R

	if stage <= 0:
		P = {}
	else:
		prereq = prereqs.get(stage, stage-1)

		P = runstage(prereq, picklepat, stagefunc,
					 force=force, prereqs=prereqs, **kwargs)

	print 'Running stage', stage
	R = stagefunc(stage, **P)
	print 'Stage', stage, 'finished'

	if update:
		if R is not None:
			P.update(R)
		R = P
		
	print 'Saving pickle', pfn
	pickle_to_file(R, pfn)
	print 'Saved', pfn
	return R
