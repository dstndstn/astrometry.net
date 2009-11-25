from xml.dom import minidom, Node
from numpy import *
from astrometry.util.pyfits_utils import *

def siap_parse_result(fn=None):
	dom1 = minidom.parse(fn)

	tables = dom1.getElementsByTagName('TABLE')
	assert(len(tables) == 1)
	table = tables[0]
	
	fields = table.getElementsByTagName('FIELD')
	print '%i fields' % len(fields)
	fieldnames = []
	fieldtypes = []
	fieldparser = []
	fieldisarray = []
	for f in fields:
		name = f.getAttribute('name').lower().replace('[]', '')
		print 'field:', name,
		ftype = f.getAttribute('datatype').lower()
		print '(%s)' % ftype
		farray = f.hasAttribute('arraysize')

		ftmap = {'int':int,
				 'double':float,
				 }

		fieldnames.append(name)
		fieldtypes.append(ftype)
		fieldparser.append(ftmap.get(ftype))
		fieldisarray.append(farray)

	data = table.getElementsByTagName('TABLEDATA')
	assert(len(data) == 1)
	data = data[0]

	rows = data.getElementsByTagName('TR')
	print '%i rows' % len(rows)

	datarows = []
	for r in rows:
		cols = r.getElementsByTagName('TD')
		assert(len(cols) == len(fields))
		datacol = []
		for c,ft,fp,fa in zip(cols, fieldtypes, fieldparser, fieldisarray):
			assert(c.firstChild)
			c = c.firstChild
			assert(c.nodeType == Node.TEXT_NODE)
			c = c.nodeValue
			datum = None
			if fa and ft in ['int','double']:
				elements = c.split(',')
				datum = array([fp(x) for x in elements])
			elif fp:
				datum = fp(c)
			else:
				datum = c
			datacol.append(datum)
		datarows.append(datacol)

	t = tabledata()
	for i,f in enumerate(fieldnames):
		t.set(f, array([r[i] for r in datarows]))
	t._length = len(datarows)

	return t
