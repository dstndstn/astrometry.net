import pyfits
import numpy
from numpy import array, isscalar, ndarray

def add_nonstructural_headers(fromhdr, tohdr):
	for card in fromhdr.ascardlist():
		if ((card.key in ['SIMPLE','XTENSION', 'BITPIX', 'END', 'PCOUNT', 'GCOUNT',
						  'TFIELDS',]) or
			card.key.startswith('NAXIS') or
			card.key.startswith('TTYPE') or
			card.key.startswith('TFORM') or
			card.key.startswith('TUNIT') or
			card.key.startswith('TDISP')):
			#print 'skipping card', card.key
			continue
		#if tohdr.has_key(card.key):
		#	#print 'skipping existing card', card.key
		#	continue
		#print 'adding card', card.key
		#tohdr.update(card.key, card.value, card.comment, before='END')
		#tohdr.ascardlist().append(
		cl = tohdr.ascardlist()
		if 'END' in cl.keys():
			i = cl.index_of('END')
		else:
			i = len(cl)
		cl.insert(i, pyfits.Card(card.key, card.value, card.comment))


class tabledata(object):

	class td_iter(object):
		def __init__(self, td):
			self.td = td
			self.i = 0
		def __iter__(self):
			return self
		def next(self):
			if self.i >= len(self.td):
				raise StopIteration
			X = self.td[self.i]
			self.i += 1
			return X


	def __init__(self, header=None):
		self._length = 0
		self._header = header
		self._columns = []
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
		#print 'set', name, 'to', val
		if (self._length == 0) and (not (name.startswith('_'))) and hasattr(val, '__len__') and len(val) != 0 and type(val) != str:
			self._length = len(val)
		if hasattr(self, '_columns') and not name in self._columns:
			self._columns.append(name)
	def set(self, name, val):
		self.__setattr__(name, val)
	def getcolumn(self, name):
		return self.__dict__[name.lower()]

	# Returns the list of column names, as they were ordered in the input FITS or text table.
	def get_columns(self):
		return self._columns
	# Returns the original FITS header.
	def get_header(self):
		return self._header

	def columns(self):
		return [k for k in self.__dict__.keys() if not k.startswith('_')]
	def __len__(self):
		return self._length
	def delete_column(self, c):
		del self.__dict__[c]
		self._columns.remove(c)
	def __getitem__(self, I):
		rtn = tabledata()
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			try:
				rtn.set(name, val[I])
			#except Exception as e:
			except Exception:
				# HACK -- emulate numpy's boolean and int array slicing...
				ok = False
				if type(I) == numpy.ndarray and hasattr(I, 'dtype') and I.dtype == 'bool':
					rtn.set(name, [val[i] for i,b in enumerate(I) if b])
					ok = True
				if type(I) == numpy.ndarray and hasattr(I, 'dtype') and I.dtype == 'int':
					rtn.set(name, [val[i] for i in I])
					ok = True
				if len(I) == 0:
					rtn.set(name, [])
					ok = True
				if not ok:
					print 'Error in slicing an astrometry.net.pyfits_utils.table_data object:'
					#print '  -->', e
					print 'While setting member:', name
					print ' by taking elements:', I
					print ' from', val
					print ' type:', type(val)
					if hasattr(val, 'shape'):
						print ' shape:', val.shape
					if hasattr(I, 'shape'):
						print ' index shape:', I.shape
					print 'index type:', type(I)
					if hasattr(I, 'dtype'):
						print '  index dtype:', I.dtype
					print 'my length:', self._length
					raise Exception('error in fits_table indexing')


			if isscalar(I):
				rtn._length = 1
			else:
				rtn._length = len(getattr(rtn, name))
		rtn._header = self._header
		if hasattr(self, '_columns'):
			rtn._columns = self._columns
		return rtn
	def __iter__(self):
		return tabledata.td_iter(self)

	def append(self, X):
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			newX = numpy.append(val, X.getcolumn(name), axis=0)
			self.set(name, newX)
			self._length = len(newX)

	def write_to(self, fn, columns=None, header='default', primheader=None):
		if columns is None and hasattr(self, '_columns'):
			columns = self._columns
		T = pyfits.new_table(self.to_fits_columns(columns))
		if header == 'default':
			header = self._header
		if header is not None:
			add_nonstructural_headers(header, T.header)
		if primheader is not None:
			P = pyfits.PrimaryHDU()
			add_nonstructural_headers(primheader, P.header)
			pyfits.HDUList([P, T]).writeto(fn, clobber=True)
		else:
			T.writeto(fn, clobber=True)

	writeto = write_to

	def to_fits_columns(self, columns=None):
		cols = []

		fmap = {numpy.float64:'D',
				numpy.float32:'E',
				numpy.int32:'J',
				numpy.int64:'K',
				numpy.uint8:'B', #
				numpy.int16:'I',
				#numpy.bool:'X',
				#numpy.bool_:'X',
				numpy.bool:'L',
				numpy.bool_:'L',
				numpy.string_:'A',
				}

		if columns is None:
			columns = self.__dict__.keys()
				
		for name in columns:
			if name.startswith('_'):
				continue
			if not name in self.__dict__:
				continue
			val = self.__dict__.get(name)
			#print 'col', name, 'type', val.dtype, 'descr', val.dtype.descr
			#print repr(val.dtype)
			#print val.dtype.type
			#print repr(val.dtype.type)
			#print val.shape
			#print val.size
			#print val.itemsize
			fitstype = fmap.get(val.dtype.type, 'D')

			if fitstype == 'X':
				# pack bits...
				pass
			if len(val.shape) > 1:
				fitstype = '%i%s' % (val.shape[1], fitstype)
			elif fitstype == 'A' and val.itemsize > 1:
				# strings
				fitstype = '%i%s' % (val.itemsize, fitstype)
			else:
				fitstype = '1'+fitstype
			#print 'fits type', fitstype
			col = pyfits.Column(name=name, array=val, format=fitstype)
			cols.append(col)
			#print 'fits type', fitstype, 'column', col
			#print repr(col)
			#print 'col', name, ': data length:', val.shape
		return cols
		

def table_fields(dataorfn, rows=None, hdunum=1, header='default'):
	pf = None
	hdr = None
	if isinstance(dataorfn, str):
		pf = pyfits.open(dataorfn)
		data = pf[hdunum].data
		if header == 'default':
			hdr = pf[hdunum].header
	else:
		data = dataorfn

	if data is None:
		return None
	fields = tabledata(header=hdr)
	colnames = data.dtype.names
	for c in colnames:
		col = data.field(c)
		if rows is not None:
			col = col[rows]
		fields.set(c.lower(), col)
	fields._length = len(data)
	fields._columns = [c.lower() for c in colnames]
	if pf:
		pf.close()
	return fields

fits_table = table_fields

# ultra-brittle text table parsing.
def text_table_fields(forfn, text=None, skiplines=0, split=None, trycsv=True, maxcols=None):
	if text is None:
		f = None
		if isinstance(forfn, str):
			f = open(forfn)
			data = f.read()
			f.close()
		else:
			data = forfn.read()
	else:
		data = text
	txtrows = data.split('\n')

	txtrows = txtrows[skiplines:]

	# column names are in the first (un-skipped) line.
	txt = txtrows.pop(0)
	header = txt
	if header[0] == '#':
		header = header[1:]
	header = header.split()
	if len(header) == 0:
		raise Exception('Expected to find column names in the first row of text; got \"%s\".' % txt)
	#assert(len(header) >= 1)
	if trycsv and (split is None) and (len(header) == 1) and (',' in header[0]):
		# try CSV
		header = header[0].split(',')
	colnames = header

	fields = tabledata()
	txtrows = [r for r in txtrows if not r.startswith('#')]
	coldata = [[] for x in colnames]
	for i,r in enumerate(txtrows):
		if maxcols is not None:
			r = r[:maxcols]
		if split is None:
			cols = r.split()
		else:
			cols = r.split(split)
		if len(cols) == 0:
			continue
		if trycsv and (split is None) and (len(cols) != len(colnames)) and (',' in r):
			# try to parse as CSV.
			cols = r.split(',')
			
		if len(cols) != len(colnames):
			raise Exception('Expected to find %i columns of data to match headers (%s) in row %i; got %i\n  "%s"' % (len(colnames), ', '.join(colnames), i, len(cols), r))
		#assert(len(cols) == len(colnames))
		for i,c in enumerate(cols):
			coldata[i].append(c)

	for i,col in enumerate(coldata):
		isint = True
		isfloat = True
		for x in col:
			try:
				float(x)
			except:
				isfloat = False
				#isint = False
				#break
			try:
				int(x, 0)
			except:
				isint = False
				#break
			if not isint and not isfloat:
				break
		if isint:
			isfloat = False

		if isint:
			vals = [int(x, 0) for x in col]
		elif isfloat:
			vals = [float(x) for x in col]
		else:
			vals = col

		fields.set(colnames[i].lower(), array(vals))
		fields._length = len(vals)

	fields._columns = [c.lower() for c in colnames]

	return fields
