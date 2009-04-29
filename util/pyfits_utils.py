import pyfits

class tabledata(object):
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
	def set(self, name,val):
		self.__setattr__(name, val)

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

	if pf:
		pf.close()
	return fields
