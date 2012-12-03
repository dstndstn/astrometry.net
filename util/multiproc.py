import multiprocessing

class FakeAsyncResult(object):
	def __init__(self, X):
		self.X = X
	def wait(self, *a):
		pass
	def get(self, *a):
		return self.X
	def ready(self):
		return True
	def successful(self):
		return True

#class PostprocessedAsyncResult(object):
#	pass

class funcwrapper(object):
	def __init__(self, func):
		self.func = func
	def __call__(self, *X):
		#print 'Trying to call', self.func
		#print 'with args', X
		try:
			return self.func(*X)
		except:
			import traceback
			print 'Exception while calling your function:'
			print '	 params:', X
			print '	 exception:'
			traceback.print_exc()
			raise

class memberfuncwrapper(object):
	def __init__(self, obj, funcname):
		self.obj = obj
		self.funcname = funcname
	def __call__(self, *X):
		func = self.obj.getattr(self.funcname)
		#print 'Trying to call', self.func
		#print 'with args', X
		try:
			return func(self.obj, *X)
		except:
			import traceback
			print 'Exception while calling your function:'
			print '  object:', self.obj
			print '  member function:', self.funcname
			print '  ', func
			print '	 params:', X
			print '	 exception:'
			traceback.print_exc()
			raise



class multiproc(object):
	def __init__(self, nthreads=1, init=None, initargs=None,
				 map_chunksize=1, pool=None, wrap_all=False):
		self.wrap_all = wrap_all
		if pool is not None:
			self.pool = pool
			self.applyfunc = self.pool.apply_async
		else:
			if nthreads == 1:
				self.pool = None
				# self.map = map
				self.applyfunc = apply
				if init is not None:
					init(*initargs)
			else:
				self.pool = multiprocessing.Pool(nthreads, init, initargs)
				# self.map = self.pool.map
				self.applyfunc = self.pool.apply_async
		self.async_results = []
		self.map_chunksize = map_chunksize

	def map(self, f, args, chunksize=None, wrap=False):
		cs = chunksize
		if cs is None:
			cs = self.map_chunksize
		if self.pool:
			if wrap or self.wrap_all:
				f = funcwrapper(f)
			#print 'pool.map: f', f
			#print 'args', args
			#print 'cs', cs
			return self.pool.map(f, args, cs)
		return map(f, args)

	def map_async(self, func, iterable, wrap=False):
		if self.pool is None:
			return FakeAsyncResult(map(func, iterable))
		if wrap or self.wrap_all:
			return self.pool.map_async(funcwrapper(func), iterable)
		return self.pool.map_async(func, iterable)
		
	def apply(self, f, args, kwargs={}):
		if self.pool is None:
			return apply(f, args, kwargs)
		res = self.applyfunc(f, args, kwargs)
		self.async_results.append(res)
		return res

	def waitforall(self):
		print 'Waiting for async results to finish...'
		for r in self.async_results:
			print '  waiting for', r
			r.wait()
		print 'all done'
		self.async_results = []
