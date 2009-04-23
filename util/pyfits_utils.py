import pyfits

class tabledata(object):
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
	def set(self, name,val):
		self.__setattr__(name, val)

def table_fields(data):
	colnames = data.dtype.names
	fields = tabledata()
	for c in colnames:
		fields.set(c.lower(), data.field(c))
	return fields
