import sys

from urllib import urlencode
from urllib2 import urlopen

from astrometry.util.file import *

if __name__ == '__main__':
    axy = read_file(sys.argv[1])
    # FIXME
    axy = axy.encode('base64_codec')
    data = urlencode({ 'axy': axy, 'jobid':'123456' })
    url = sys.argv[2]

    f = urlopen(url, data)
    response = f.read()
    f.close()
    
    print response
    
