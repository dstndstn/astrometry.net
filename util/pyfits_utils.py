import pyfits

class tabledata(object):
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
	def set(self, name,val):
		self.__setattr__(name, val)

def table_fields(data):
	colnames = data.dtype.names
	#fields = {}
	#fields = object()
	fields = tabledata()
	for c in colnames:
		#print 'Adding field', c.lower()
		#fields[c.lower()] = data.field(c)
		#fields.__dict__[c.lower()] = data.field(c)
		#object.__setattr__(fields, c.lower(), data.field(c))
		fields.set(c.lower(), data.field(c))
	return fields
