import pyfits

def table_fields(data):
	colnames = data.dtype.names
	#fields = {}
	fields = object()
	for c in colnames:
		fields[c.tolower()] = data.field(c)
	return fields
