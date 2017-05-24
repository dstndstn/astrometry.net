# Copyright 2011 David W. Hogg.
# All rights reserved.

# BUGS:
# - Brittle code; must be run from directory client/examples; dies if APOD reformats urls or html.
# - Runs client using os.system() instead of importing client and executing it; see if False block at end.
from __future__ import print_function

import re
import os
import sys
import urllib as url

from astrometry.net.client import Client

def apod_baseurl():
    return "http://apod.nasa.gov/apod/"

def apod_url(month, day, year):
    datestr = "%02d%02d%02d" % ((year % 100), month, day)
    return "%sap%s.html" % (apod_baseurl(), datestr)

def get_apod_image_url(aurl):
    f = url.urlopen(aurl)
    for line in f:
        if re.search(r'[Ss][Rr][Cc]', line) is not None:
            return apod_baseurl()+re.split(r'^.*[Ss][Rs][Cc]=\"(.+)\".*', line)[1]
    f.close
    return None

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('--server', dest='server', default='http://supernova.astrometry.net/api/',
                      help='Set server base URL (eg, http://nova.astrometry.net/api/)')
    parser.add_option('--apikey', '-k', dest='apikey',
                      help='API key for Astrometry.net web service; if not given will check AN_API_KEY environment variable')
    opt,args = parser.parse_args()
    if opt.apikey is None:
        # try the environment
        opt.apikey = os.environ.get('AN_API_KEY', None)
    if opt.apikey is None:
        parser.print_help()
        print()
        print('You must either specify --apikey or set AN_API_KEY')
        sys.exit(-1)

    useclient = True
    if useclient:
        client = Client(apiurl=opt.server)
        client.login(opt.apikey)

    for year in range(1996, 2013):
        for month in range(1, 13):
            print("apod.py __main__: working on month %d-%02d" % (year, month))
            for day in range(1, 32):
                iurl = get_apod_image_url(apod_url(month, day, year))
                if iurl is None:
                    continue
                if useclient:
                    client.url_upload(iurl)
                    print(client.submission_images(1))
                else:
                    cmd = "python ../client.py --server %s --apikey %s --urlupload \"%s\"" % (opt.server, opt.apikey, iurl)
                    print(cmd)
                    os.system(cmd)
