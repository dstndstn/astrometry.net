import simplejson
import re

lines = open('brightstars-data.c').readlines()
l = ''.join(lines[3:-2]).replace('{','[').replace('}',']')
#rex = re.compile(r'\\x(..)\\x(..)')
#l = rex.sub(l, '\\u\1\2')
l = re.sub(r'\\x(..)\\x(..)""', r'\u\1\2 ', l)
l = '[' + l + '0 ]'
#print l

j = simplejson.loads(l)
j = j[:-1]
print j

nm, nm2, rr, dd = [],[],[],[]
for n1,n2,r,d,mag in j:
    nm.append(simplejson.dumps(n1))
    nm2.append(n2)
    rr.append(r)
    dd.append(d)

from astrometry.util.fits import *
import numpy as np

T = tabledata()
T.name1 = np.array(nm)
T.name2 = np.array(nm2)
T.ra = np.array(rr)
T.dec = np.array(dd)
T.writeto('brightstars.fits')
