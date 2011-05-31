from urllib2 import urlopen
from urllib import urlencode
from exceptions import Exception

from astrometry.net1.portal.api_util import json2python, python2json

def logverb(*msg):
    print(' '.join([str(m).decode('latin_1', 'backslashreplace') for m in msg]))

class MalformedResponse(Exception):
	pass
	#def __init__(self, msg):
	#	super(self, MaformedResponse).__init__(msg)

class RequestError(Exception):
	pass

class Client(object):

	#apiurl = 'http://edge.astrometry.net/api/'
	apiurl = 'http://oven.cosmo.fas.nyu.edu:9000/api/'

	def __init__(self):
		self.apiurl = Client.apiurl

	def get_url(self, service):
		return self.apiurl + service

	def send_request(self, service, args):
		json = python2json(args)
		logverb('Sending json:', json)
		data = urlencode({'request-json': json})
		logverb('Sending data:', data)
		url = self.get_url(service)
		logverb('Sending to URL:', url)
		f = urlopen(url, data)
		txt = f.read()
		logverb('Got json:', txt)
		result = json2python(txt)
		logverb('Got result:', result)
		stat = result.get('status')
		logverb('Got status:', stat)
		#if not stat:
		#	raise MalformedResponse('bad result')
		if stat == 'error':
			errstr = result.get('errormessage', '(none)')
			raise RequestError('server error message: ' + errstr)
		return result

	def login(self, username, password):
		args = { 'username' : username,
				 'password' : password }
		result = self.send_request('login', args)
		sess = result.get('session')
		logverb('Got session:', sess)
		if not sess:
			raise RequestError('no session in result')
		self.session = sess

	def submit_url(self, url):
		args = { 'session': self.session,
				 'url': url }
		result = self.send_request('submit_url', args)
		subid = result.get('subid')
		if not subid:
			raise RequestError('no subid in result')
		return subid

	def submission_status(self, subid):
		args = { 'session': self.session,
				 'subid': subid }
		result = self.send_request('substatus', args)
		return result

if __name__ == '__main__':
	c = Client()
	c.login('test@astrometry.net', 'password')
	url = 'http://antwrp.gsfc.nasa.gov/apod/image/0902/Lulin2_richins.jpg'
	subid = c.submit_url(url)
	print 'Submission id:', subid
	substatus = c.submission_status(subid)
	print 'Status:', substatus

