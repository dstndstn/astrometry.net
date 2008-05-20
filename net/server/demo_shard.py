import sys

from urllib import urlencode
from urllib2 import urlopen

from astrometry.util.file import *

if __name__ == '__main__':
    tar = read_file(sys.argv[1]) #.encode('base64_codec')
    data = urlencode({ 'tar': tardata, 'jobid':'123456' })
    url = sys.argv[2]

    f = urlopen(url, data)
    response = f.read()
    f.close()
    
    print response
    
