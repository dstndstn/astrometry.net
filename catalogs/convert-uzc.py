# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from astrometry.util.starutil_numpy import *
from astrometry.util.fits import *

'''
Updated Zwicky Catalog

https://www.cfa.harvard.edu/~falco/UZC/
https://www.cfa.harvard.edu/~falco/UZC/uzcJ2000.tab.gz
'''

rr,dd,nn = [],[],[]

for i,line in enumerate(open('uzcJ2000.tab')):
    if i < 20:
        continue
    #print 'line', line
    words = line.split('\t')
    #print 'words', words
    assert(len(words) == 14)

    ra = words[0]
    ra = ra[:2] + ' ' + ra[2:4] + ' ' + ra[4:]

    dec = words[1]
    dec = dec[:3] + ' ' + dec[3:5] + ' ' + dec[5:]

    #print 'ra,dec', ra,dec
    ra = hmsstring2ra(ra)
    dec = dmsstring2dec(dec)

    zname = words[8]
    
    #print 'ra,dec', ra,dec
    rr.append(ra)
    dd.append(dec)
    nn.append(zname)
    
T = tabledata()
T.ra = rr
T.dec = dd
T.zname = nn
T.writeto('uzc2000.fits')
