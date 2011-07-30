import simplejson

def json2python(json):
	try:
		return simplejson.loads(json)
	except:
		pass
	return None

python2json = simplejson.dumps

