#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

import re
import sys
import socket
import time
from urllib.request import urlopen
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import urlparse, urljoin
from os.path import basename
from optparse import OptionParser
from xml.dom import minidom

from astrometry.util.starutil_numpy import *
from astrometry.util.file import *
from astrometry.util.run_command import run_command

# args: radius in deg
# returns list of local filenames to which images were written
def get_2mass_images(ra, dec, radius=1, basefn=None, band='A'):

    formvals = {
        'type': 'at', # Atlas images (not quicklook)
        'INTERSECT': 'OVERLAPS', # Images overlapping region
        'asky': 'asky', # All-sky release
        'POS': '%g %g' % (ra,dec), # RA,Dec position
        'SIZE': '%g' % radius, # Search radius (deg)
        # scan, coadd, hem, date
        'band': band, # 'A'=All bands; alternatives: J,H,K
        }

    if basefn is None:
        #basefn = '2mass-%g-%g-' % (ra,dec)
        basefn = '2mass-'

    queryurl = 'http://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_inv'
    print('submitting form values:')
    for k,v in list(formvals.items()):
        print('  ',k,'=',v)
    print('encoded as:')
    print('  ' + urlencode(formvals))
    print()
    print('waiting for results...')
    socket.setdefaulttimeout(300)
    f = urlopen(queryurl, urlencode(formvals))
    doc = f.read()
    write_file(doc, 'res.html')

    m = re.search(r'<base href="(.*)" />', doc)
    if not m:
        raise RuntimeError('no results page: server output written to file')
    resurl = m.group(1)
    print('result base url', resurl)

    resurl += 'found.xml'

    print('requesting result url', resurl)
    res = urlopen(resurl)
    print()
    doc = res.read()
    write_file(doc, 'res2.xml')

    xmldoc = minidom.parseString(doc)

    imgs = xmldoc.getElementsByTagName('TR')
    if len(imgs) == 0:
        print('no <TR> tags found')
        return None

    urlfns = []
    for imgtag in imgs:
        print()
        if not imgtag.hasChildNodes():
            print('<TR> tag has no child node:', imgtag)
            return None
        #print 'Image:', imgtag
        tds = imgtag.getElementsByTagName('TD')
        if not len(tds):
            print('<TR> tag has no <TD> child nodes:', imgtag)
            return None

        print('Image:',  tds[0].firstChild.data)
        print('  URL:',  tds[1].firstChild.data)
        print('  Band:', tds[11].firstChild.data)
        print('  dataset (asky):', tds[18].firstChild.data)
        print('  date (yymmdd):', tds[22].firstChild.data)
        print('  hem (n/s):', tds[23].firstChild.data)
        print('  scan:', tds[24].firstChild.data)
        print('  image num:', tds[25].firstChild.data)

        url = tds[1].firstChild.data
        band = tds[11].firstChild.data
        dataset = tds[18].firstChild.data
        date = tds[22].firstChild.data
        hem = tds[23].firstChild.data
        scan = int(tds[24].firstChild.data)
        imgnum = int(tds[25].firstChild.data)

        fn = basefn + '%s_%s_%s%s%03i%04i.fits.gz' % (band, dataset, date, hem, scan, imgnum)

        urlfns.append((url,fn))

    fns = []
    for i,(url,fn) in enumerate(urlfns):
        print()
        print('Retrieving file %i of %i' % (i+1, len(urlfns)))
        print()
        # -t: num retries
        cmd = "wget -t 1 -c '%s' -O %s" % (url, fn)
        print('Running command:', cmd)

        rtn = os.system(cmd)
        # ctrl-C caught: quit, or continue?
        if os.WIFSIGNALED(rtn):
            print('wget exited with signal', os.WTERMSIG(rtn))
            break
        if os.WIFEXITED(rtn) and os.WEXITSTATUS(rtn):
            # returned non-zero.
            print('wget exited with value', os.WEXITSTATUS(rtn))
            continue

        fns.append(fn)
    return fns


if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <ra> <dec>')

    # See http://irsa.ipac.caltech.edu/applications/2MASS/IM/inventory.html#pos

    parser.add_option('-r', dest='radius', type='float', help='Search radius, in deg (default 1 deg)')
    parser.add_option('-b', dest='basefn', help='Base filename (default: 2mass-)')
    parser.add_option('-B', dest='band', help='Band (J, H, K); default: all three')
    parser.set_defaults(radius=1.0, band='A')

    (opt, args) = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        print()
        print('Got extra arguments:', args)
        sys.exit(-1)

    # parse RA,Dec.
    ra = float(args[0])
    dec = float(args[1])

    # ugh!
    opts = {}
    for k in ['radius', 'basefn', 'band']:
        opts[k] = getattr(opt, k)

    get_2mass_images(ra, dec, **opts)
