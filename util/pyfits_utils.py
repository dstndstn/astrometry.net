import pyfits
import numpy
from numpy import array, isscalar

class tabledata(object):
	def __init__(self):
		self._length = 0
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
	def set(self, name, val):
		self.__setattr__(name, val)
	def getcolumn(self, name):
		return self.__dict__[name.lower()]
	def columns(self):
		return self.__dict__.keys()
	def __len__(self):
		return self._length
	def delete_column(self, c):
		del self.__dict__[c]
	def __getitem__(self, I):
		rtn = tabledata()
		for name,val in self.__dict__.items():
			if name == '_length':
				continue
			rtn.set(name, val[I])
			if isscalar(I):
				rtn._length = 1
			else:
				rtn._length = len(val[I])
		return rtn

	def append(self, X):
		for name,val in self.__dict__.items():
			if name == '_length':
				continue
			newX = numpy.append(val[I], X.getcolumn(name))
			self.set(name, newX)
			self._length = len(newX)

	def write_to(self, fn):
		pyfits.new_table(self.to_fits_columns()).writeto(fn, clobber=True)
	def writeto(self, fn):
		return self.write_to(fn)

	def to_fits_columns(self):
		cols = []

		fmap = {numpy.float64:'D',
				numpy.float32:'E',
				numpy.int32:'J',
				numpy.uint8:'B', #
				numpy.int16:'I',
				numpy.bool:'X',
				numpy.bool_:'X',
				}
				
		for name,val in self.__dict__.items():
			if name == '_length':
				continue
			# FIXME -- format should match that of the 'val'.
			#print
			#print 'col', name, 'type', val.dtype, 'descr', val.dtype.descr
			#print repr(val.dtype)
			#print val.dtype.type
			#print repr(val.dtype.type)
			fitstype = fmap.get(val.dtype.type, 'D')

			if fitstype == 'X':
				# pack bits...
				pass

			if len(val.shape) > 1:
				fitstype = '%i%s' % (val.shape[1], fitstype)
			else:
				fitstype = '1'+fitstype

			col = pyfits.Column(name=name, array=val, format=fitstype)
			cols.append(col)
			#print 'fits type', fitstype, 'column', col
			#print repr(col)
			#print 'col', name, ': data length:', val.shape
		return cols
		

def table_fields(dataorfn, rows=None, hdunum=1):
	pf = None
	if isinstance(dataorfn, str):
		pf = pyfits.open(dataorfn)
		data = pf[hdunum].data
	else:
		data = dataorfn

	if data is None:
		return None
	fields = tabledata()
	colnames = data.dtype.names
	for c in colnames:
		col = data.field(c)
		if rows is not None:
			col = col[rows]
		fields.set(c.lower(), col)
	fields._length = len(data)
	if pf:
		pf.close()
	return fields

# ultra-brittle text table parsing.
def text_table_fields(forfn, text=None, skiplines=0, split=None):
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
	header = txtrows.pop(0)
	header = header.split()
	if header[0] == '#':
		header = header[1:]
	assert(len(header) >= 1)
	colnames = header[1:]

	fields = tabledata()
	txtrows = [r for r in txtrows if not r.startswith('#')]
	coldata = [[] for x in colnames]
	for r in txtrows:
		if split is None:
			cols = r.split()
		else:
			cols = r.split(split)
		if len(cols) == 0:
			continue
		if split is None and len(cols) != len(colnames) and ',' in r:
			# try to parse as CSV.
			cols = r.split(',')
			
		assert(len(cols) == len(colnames))
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
				break
			try:
				int(x, 0)
			except:
				isint = False
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

	return fields
