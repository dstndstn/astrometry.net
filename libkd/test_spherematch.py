import spherematch
from numpy import *

x = array([[1.,2.,3.],[4.,5.,6.]])

print 'x =', x

y = spherematch.spherematch_info(x)

print 'y =', y
