import pyfits

class tabledata(object):
	def __init__(self):
		self._length = 0
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
	def set(self, name,val):
		self.__setattr__(name, val)
	def getcolumn(self, name):
		return self.__dict__[name.lower()]
	def __len__(self):
		return self._length

def table_fields(dataorfn):
	pf = None
	if isinstance(dataorfn, str):
		pf = pyfits.open(dataorfn)
		data = pf[1].data
	else:
		data = dataorfn

	colnames = data.dtype.names
	fields = tabledata()
	for c in colnames:
		fields.set(c.lower(), data.field(c))
	fields._length = len(data)
	if pf:
		pf.close()
	return fields
