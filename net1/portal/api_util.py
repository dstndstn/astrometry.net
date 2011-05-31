import simplejson

def json2python(json):
	try:
		return simplejson.loads(json)
	except:
		pass
	return None

def python2json(py):
	return simplejson.dumps(py)

