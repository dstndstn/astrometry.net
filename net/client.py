import os
import sys
from urllib2 import urlopen
from urllib2 import Request
from urllib2 import HTTPError
from urllib import urlencode
from exceptions import Exception
from email.mime.multipart import MIMEMultipart

from email.mime.base import MIMEBase
from email.mime.application  import MIMEApplication

from email.encoders import encode_noop

from astrometry.net.api_util import json2python, python2json

class MalformedResponse(Exception):
    pass
class RequestError(Exception):
    pass


class Client(object):

    def __init__(self,
                 apiurl = 'http://oven.cosmo.fas.nyu.edu:9002/api/'):
        self.session = None
        self.apiurl = apiurl

    def get_url(self, service):
        return self.apiurl + service

    def send_request(self, service, args, file_args=None):
        '''
        service: string
        args: dict
        '''
        if self.session is not None:
            args.update({ 'session' : self.session })
        print 'Python:', args
        json = python2json(args)
        print 'Sending json:', json
        data = urlencode({'request-json': json})
        print 'Sending data:', data
        url = self.get_url(service)
        print 'Sending to URL:', url
        #f = urlopen(url, data)

        m = []
        m1 = MIMEBase('text', 'plain')
        m1.add_header('Content-disposition', 'form-data; name="request-json"')
        m1.set_payload(json)
        m += [m1]

        if file_args is not None:
            m2 = MIMEApplication(file_args[1],'octet-stream',encode_noop)
            m2.add_header('Content-disposition', 'form-data; name="file"; filename="%s"' % file_args[0])
            m += [m2]

        mp = MIMEMultipart('form-data', None, m)
        
        data = mp.as_string().split('\n')
        for i,d in enumerate(data):
            if len(d) == 0:
                break
        data = data[i+1:]
        print 'data:', data

        request = Request(url=url,
                  headers={'Content-type': mp.get('Content-type')},
                  data='\r\n'.join(data))

        try:
            f = urlopen(request)
            txt = f.read()
            print 'Got json:', txt
            result = json2python(txt)
            print 'Got result:', result
            stat = result.get('status')
            print 'Got status:', stat
            if stat == 'error':
                errstr = result.get('errormessage', '(none)')
                raise RequestError('server error message: ' + errstr)
            return result
        except HTTPError as e:
            print 'HTTPError', e
            txt = e.read()
            open('err.html', 'wb').write(txt)

    def login(self, apikey):
        args = { 'apikey' : apikey }
        result = self.send_request('login', args)
        sess = result.get('session')
        print 'Got session:', sess
        if not sess:
            raise RequestError('no session in result')
        self.session = sess

    def upload(self, fn):
        try:
            f = open(fn)
            result = self.send_request('upload', {}, (fn, f.read()))
        except IOError:
            print 'File %s does not exist' % fn     
            raise

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--upload', '-u', dest='upload', help='Upload a file')
    parser.add_option('--apikey', '-k', dest='apikey',
                      help='API key for Astrometry.net web service; if not given will check AN_API_KEY environment variable')
    opt,args = parser.parse_args()

    if opt.apikey is None:
        # try the environment
        opt.apikey = os.environ.get('AN_API_KEY', None)
    if opt.apikey is None:
        parser.print_help()
        print
        print 'You must either specify --apikey or set AN_API_KEY'
        sys.exit(-1)

    c = Client()
    c.login(opt.apikey)

    if opt.upload:
        c.upload(opt.upload)
        
