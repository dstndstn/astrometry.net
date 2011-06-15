import os
import sys
import base64
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

    def overlay_plot(self, service, outfn, wcsfn, wcsext=0):
        from astrometry.util import util as anutil
        wcs = anutil.Tan(wcsfn, wcsext)
        params = dict(crval1 = wcs.crval[0], crval2 = wcs.crval[1],
                      crpix1 = wcs.crpix[0], crpix2 = wcs.crpix[1],
                      cd11 = wcs.cd[0], cd12 = wcs.cd[1],
                      cd21 = wcs.cd[2], cd22 = wcs.cd[3],
                      imagew = wcs.imagew, imageh = wcs.imageh)
        result = self.send_request(service, {'wcs':params})
        print 'Result:', result
        plotdata = result['plot']
        plotdata = base64.b64decode(plotdata)
        open(outfn, 'wb').write(plotdata)

    def sdss_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('sdss_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def galex_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('galex_image_for_wcs', 'galex.png',
                                 wcsfn, wcsext)

    
if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--server', dest='server',
                      help='Set server base URL (eg, http://localhost:8000/api/)')
    parser.add_option('--apikey', '-k', dest='apikey',
                      help='API key for Astrometry.net web service; if not given will check AN_API_KEY environment variable')
    parser.add_option('--upload', '-u', dest='upload', help='Upload a file')
    parser.add_option('--sdss', dest='sdss_wcs', nargs=2, help='Plot SDSS image for the given WCS file; write plot to given PNG filename')
    parser.add_option('--galex', dest='galex_wcs', nargs=2, help='Plot GALEX image for the given WCS file; write plot to given JPG filename')
    opt,args = parser.parse_args()

    if opt.apikey is None:
        # try the environment
        opt.apikey = os.environ.get('AN_API_KEY', None)
    if opt.apikey is None:
        parser.print_help()
        print
        print 'You must either specify --apikey or set AN_API_KEY'
        sys.exit(-1)

    args = {}
    if opt.server:
        args['apiurl'] = opt.server
    c = Client(**args)
    c.login(opt.apikey)

    if opt.upload:
        c.upload(opt.upload)
    if opt.sdss_wcs:
        (wcsfn, outfn) = opt.sdss_wcs
        c.sdss_plot(outfn, wcsfn)
    if opt.galex_wcs:
        (wcsfn, outfn) = opt.galex_wcs
        c.galex_plot(outfn, wcsfn)

