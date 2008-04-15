import urllib
import tempfile
import os

class W3CValidator:
    def __init__(self, url=None):
        if url is None:
            self.url = 'http://validator.w3.org/'
        else:
            self.url = url

    def validateText(self, text):
        postdata = urllib.urlencode({ 'fragment' : text })
        (f, filename) = tempfile.mkstemp('.html', 'w3cvalid-')
        os.close(f)
        (fn, headers) = urllib.urlretrieve(self.url, filename, None, postdata)
        key = 'X-W3C-Validator-Status'
        if not (key in headers):
            print 'W3CValidator: no key "%s" in response headers.  Results saved in file %s' % (key, filename)
            print 'Headers:'
            for (k,v) in headers.items():
                print '  ', k, '=', v
            return False
        if headers[key] == 'Valid':
            os.remove(filename)
            return True
        print 'W3CValidator: %s = %s.  Results saved in file %s' % (key, headers[key], filename)
        print 'Headers:'
        for (k,v) in headers.items():
            print '  ', k, '=', v
        return False


if __name__ == '__main__':
    v = W3CValidator('http://oven.cosmo.fas.nyu.edu:8888/w3c-markup-validator/check')
    v.validateText('<html></html>')

