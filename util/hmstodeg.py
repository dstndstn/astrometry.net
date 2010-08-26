#! /usr/bin/env python
import sys

from astrometry.util.starutil_numpy import *

if __name__ == '__main__':
	args = sys.argv[1:]
	if len(args) != 2:
		print 'Usage: %s <h:m:s> <d:m:s>' % sys.argv[0]
		sys.exit(-1)

	hms = args[0]
	dms = args[1]

	ra = hmsstring2ra(hms)
	rastr = ra2hmsstring(ra)

	dec = dmsstring2dec(dms)
	decstr = dec2dmsstring(dec)

	print '            %-20s   %-20s' % ('RA', 'Dec')
	print 'in:         %-20s   %-20s' % (hms, dms)
	print 'parsed as:  %-20s   %-20s' % (rastr, decstr)
	print 'deg:        %-20f   %-20f' % (ra, dec)

	
