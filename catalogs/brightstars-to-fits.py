# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import simplejson
import re

lines = open('brightstars-data.c').readlines()
l = ''.join(lines[3:-2]).replace('{','[').replace('}',']')

def replace_unicode(match):
    c1 = match.group(1)
    c2 = match.group(2)
    #s = eval('\\x%s\\x%s'
    s = '"\\x%s\\x%s"' % (c1, c2)
    #print 's', s
    s = eval(s)
    #print 's', s
    d = s.decode('utf8')
    return d + ' '

l = re.sub(r'\\x(..)\\x(..)""', replace_unicode, l)
#l = re.sub(r'\\x(..)\\x(..)""', r'\u\1\2 ', l)
#l = re.sub(r'\\x(..)\\x(..)""', r'\x\1\x\2 ', l)
#l = l.decode('utf8')

l = '[' + l + '0 ]'
print(l)

j = simplejson.loads(l)
j = j[:-1]
print(j)

nm, nm2, rr, dd = [],[],[],[]
vmag = []
for n1,n2,r,d,mag in j:
    #print 'n1', n1
    j = simplejson.dumps(n1)
    #print 'json:', j
    #j = str(j)
    #print '  ->', j
    nm.append(j.replace('"', ''))
    nm2.append(str(n2))
    vmag.append(float(mag))
    rr.append(r)
    dd.append(d)

from astrometry.util.fits import *
import numpy as np

T = tabledata()
T.name1 = np.array(nm)
T.name2 = np.array(nm2)
T.vmag = np.array(vmag)
T.ra = np.array(rr)
T.dec = np.array(dd)
T.about()
T.writeto('brightstars.fits')
