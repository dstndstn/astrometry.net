#! /usr/bin/env python

import sys

from astrometry.util.starutil_numpy import *

if __name__ == '__main__':
	args = sys.argv[1:]
	if len(args) != 2:
		print 'Usage: %s <ra> <dec>' % sys.argv[0]
		sys.exit(-1)

	ra  = float(args[0])
	dec = float(args[1])

	rastr  = ra2hmsstring(ra).replace(' ','h',1).replace(' ','m',1).replace(' ','s',1)
	decstr = dec2dmsstring(dec).replace(' ','d',1).replace(' ','m',1).replace(' ','s',1)

	print '            %-20s   %-20s' % ('RA', 'Dec')
	print 'in:         %-20f   %-20f' % (ra, dec)
	print 'out:        %-20s   %-20s' % (ra2hmsstring(ra), dec2dmsstring(dec))
	print 'out:        %-20s   %-20s' % (ra2hmsstring(ra,':'), dec2dmsstring(dec, ':'))
	print 'out:        %-20s   %-20s' % (rastr, decstr)

	
