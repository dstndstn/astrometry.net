#! /usr/bin/env python
import os
import sys
import time
from optparse import *
from math import *

from astrometry.util import casjobs
from astrometry.util.starutil_numpy import *

# override numpy.random
import random

def sdss_cas_get(username, password, ra, dec, radius, fn, band=None, maxmag=None, delete=True, mydbname=None, dr=None):

	casjobs.setup_cookies()

	if dr is None:
		dr = 'dr7'
	cas = casjobs.get_known_servers()[dr]
	cas.login(username, password)

	if mydbname is None:
		az = [chr(i) for i in range(ord('A'), ord('Z')+1)]
		sam = random.sample(az, 10)
		mydbname = 'cat_' + ''.join(sam)# random.sample(az, 10))

	if band is not None and maxmag is not None:
		magcut = 'and %s <= %g' % (band, maxmag)
	else:
		magcut = ''

	# It's MUCH faster to pre-compute the bounding RA,Dec box in addition
	# to the fDistanceArcMinEq call.
	ramin = max(0, ra - radius / cos(deg2rad(dec)))
	ramax = min(360, ra + radius / cos(deg2rad(dec)))
	decmin = dec - radius
	decmax = dec + radius

	sql = ' '.join(['select ra,dec,raErr,decErr,flags,u,g,r,i,z,'
					'err_u,err_g,err_r,err_i,err_z',
					('into mydb.%s' % mydbname),
					'from PhotoPrimary',
					('where ra >= %g and ra <= %g and dec >= %g and dec <= %g' %
					 (ramin, ramax, decmin, decmax)),
					('and dbo.fDistanceArcMinEq(%g, %g, ra, dec) <= %g' %
					 (ra, dec, deg2arcmin(radius))),
					magcut
					])
	print 'Submitting SQL:'
	print sql
	
	jobid = cas.submit_query(sql)
	print 'jobid', jobid

	while True:
		jobstatus = cas.get_job_status(jobid)
		print 'Job id', jobid, 'is', jobstatus
		if jobstatus in ['Finished', 'Failed', 'Cancelled']:
			break
		print 'Waiting...'
		time.sleep(10)

	print 'Requesting output...'
	cas.output_and_download(mydbname, fn, dodelete=delete)

					  
if __name__ == '__main__':
	parser = OptionParser(usage='%prog [options] <ra> <dec> <output-filename>')

	parser.add_option('-r', dest='radius', type='float', help='Search radius, in deg (default 1 deg)')
	parser.add_option('-b', dest='band', help='Band to use for mag and flag cuts (default: r, options: u,g,r,i,z')
	parser.add_option('-m', dest='maxmag', type='float', help='Maximum magnitude cut')
	parser.add_option('-u', dest='username', help='SDSS CasJobs username (default: from $SDSS_CAS_USER)')
	parser.add_option('-p', dest='password', help='SDSS CasJobs password (default: from $SDSS_CAS_PASS)')
	parser.add_option('-R', dest='dr', type='string', help='CAS server to use ("dr7" or "dr8")')
	parser.set_defaults(radius=1.0, band='r', maxmag=20., username=None, password=None, dr=None)

	(opt, args) = parser.parse_args()
	if len(args) != 3:
		parser.print_help()
		print
		print 'Got extra arguments:', args
		sys.exit(-1)

	if opt.username is None:
		opt.username = os.environ['SDSS_CAS_USER']
	if opt.password is None:
		opt.password = os.environ['SDSS_CAS_PASS']
	if opt.username is None or opt.password is None:
		print 'Need CAS username and password.'
		sys.exit(-1)

	# parse RA,Dec.
	ra = float(args[0])
	dec = float(args[1])
	outfn = args[2]

	sdss_cas_get(opt.username, opt.password, ra, dec, opt.radius, outfn,
				 opt.band, opt.maxmag, dr=opt.dr)

