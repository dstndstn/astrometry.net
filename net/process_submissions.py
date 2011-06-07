#! /usr/bin/env python

import os
import sys
import tempfile
import traceback
from urlparse import urlparse
import logging
import urllib
import shutil
import multiprocessing
import time
import re

import logging
logging.basicConfig(format='%(message)s',
					level=logging.DEBUG)

from astrometry.util import image2pnm
from astrometry.util.filetype import filetype_short
from astrometry.util.run_command import run_command

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
#import astrometry.net.settings as settings
import settings
from astrometry.net.models import *
from log import *



def is_tarball(fn):
	logmsg('is_tarball: %s' % fn)
	types = filetype_short(fn)
	logmsg('filetypes:', types)
	for t in types:
		if t.startswith('POSIX tar archive'):
			return True
	return False

def get_tarball_files(fn):
	# create temp dir to extract tarfile.
	tempdir = tempfile.mkdtemp()
	cmd = 'tar xvf %s -C %s' % (fn, tempdir)
	#userlog('Extracting tarball...')
	(rtn, out, err) = run_command(cmd)
	if rtn:
		#userlog('Failed to un-tar file:\n' + err)
		#bailout(submission, 'failed to extract tar file')
		print 'failed to extract tar file'
		return None
	fns = out.strip('\n').split('\n')

	validpaths = []
	for fn in fns:
		path = os.path.join(tempdir, fn)
		logmsg('Path "%s"' % path)
		if not os.path.exists(path):
			logmsg('Path "%s" does not exist.' % path)
			continue
		if os.path.islink(path):
			logmsg('Path "%s" is a symlink.' % path)
			continue
		if os.path.isfile(path):
			validpaths.append(path)
		else:
			logmsg('Path "%s" is not a file.' % path)

	if len(validpaths) == 0:
		#userlog('Tar file contains no regular files.')
		#bailout(submission, "tar file contains no regular files.")
		#return -1
		logmsg('No real files in tar file')
		return None

	logmsg('Got %i paths.' % len(validpaths))
	return validpaths

def run_pnmfile(fn):
	cmd = 'pnmfile %s' % fn
	(filein, fileout) = os.popen2(cmd)
	filein.close()
	out = fileout.read().strip()
	logmsg('pnmfile output: ' + out)
	pat = re.compile(r'P(?P<pnmtype>[BGP])M .*, (?P<width>\d*) by (?P<height>\d*) *maxval (?P<maxval>\d*)')
	match = pat.search(out)
	if not match:
		logmsg('No match.')
		return None
	w = int(match.group('width'))
	h = int(match.group('height'))
	pnmtype = match.group('pnmtype')
	maxval = int(match.group('maxval'))
	logmsg('Type %s, w %i, h %i, maxval %i' % (pnmtype, w, h, maxval))
	return (w, h, pnmtype, maxval)



def dosub(sub):
	### FIXME
	sshconfig = 'an-test'

	print 'Processing submission:', sub
	sub.set_processing_started()
	sub.save()
	origname = None
	if sub.disk_file is None:
		print 'Retrieving URL', sub.url
		(fn, headers) = urllib.urlretrieve(sub.url)
		print 'Wrote to file', fn
		df = DiskFile.fromFile(fn)
		# Try to split the URL into a filename component and save it
		p = urlparse(submission.url)
		p = p.path
		if p:
			s = p.split('/')
			origname = s[-1]
			sub.orig_filename = origname
		df.save()
		sub.file = df
		sub.save()
	else:
		origname = sub.original_filename

	# compressed .gz?
	df = sub.disk_file
	fn = df.get_path()
	f,tmpfn = tempfile.mkstemp()
	os.close(f)
	comp = image2pnm.uncompress_file(fn, tmpfn)
	if comp:
		print 'Input file compression: %s' % comp
		fn = tmpfn

	# This is sort of crazy -- look at python's 'gzip' and 'tarfile' modules.
	'''
	if is_tarball(fn):
		logmsg('file is tarball.')
		fns = get_tarball_files(fn)
		if fns is None:
			return

		for fn in fns:
			df = DiskFile.for_file(fn)
			df.save()
			logmsg('New diskfile ' + df)
		shutil.rmtree(tempdir)
		return True
	'''

	# create Image object
	# FIXME -- should check if "df" already has an Image!

	# FIXME -- move this code to Image, probably
	# Convert file to pnm and find its size.
	f,pnmfn = tempfile.mkstemp()
	os.close(f)
	logmsg('Converting %s to %s...\n' % (fn, pnmfn))
	(filetype, errstr) = image2pnm.image2pnm(fn, pnmfn)
	if errstr:
		logmsg('Error converting image file: %s' % errstr)
		#df.filetype = filetype
		#return fullfn
		return
	x = run_pnmfile(pnmfn)
	if x is None:
		print 'couldn\'t find image file size'
		return
	(w, h, pnmtype, maxval) = x
	logmsg('Type %s, w %i, h %i' % (pnmtype, w, h))
	
	img = Image(disk_file=df, width=w, height=h)
	img.save()

	# create UserImage object.
	uimg = UserImage(submission=sub, image=img,
					 original_file_name=origname)
	uimg.save()

	# run solve-field or whatever.

	job = Job(user_image=uimg)
	job.set_start_time()
	job.save()

	# create FITS image
	# run image2xy
	#infn = convert(job, 'fitsimg')
	#cmd = 'image2xy -v %s-o %s %s >> %s 2>&1' % (extraargs, fullfn, infn, sxylog)
	#run_convert_command(cmd)

	f,axyfn = tempfile.mkstemp()
	os.close(f)

	slo,shi = sub.get_scale_bounds()
	
	axyargs = {
		'--out': axyfn,
		#'--width': img.width,
		#'--height': img.height,
		# Aww yah
		# job.submission.userimage.image.file
		'--image': df.get_path(),
		'--scale-low': slo,
		'--scale-high': shi,
		'--scale-units': sub.scale_units,
		#'--pixel-error':,
		# --use-sextractor
		# --ra, --dec, --radius
		# --invert
		'--cancel': 'none',
		'--solved': 'none',
		'--match': 'none',
		'--rdls': 'none',
		'--corr': 'none',
		# -g / --guess-scale: try to guess the image scale from the FITS headers
		# --crpix-center: set the WCS reference point to the image center
		# --crpix-x <pix>: set the WCS reference point to the given position
		# --crpix-y <pix>: set the WCS reference point to the given position
		# -T / --no-tweak: don't fine-tune WCS by computing a SIP polynomial
		# -t / --tweak-order <int>: polynomial order of SIP WCS corrections
		# -w / --width <pixels>: specify the field width
		# -e / --height <pixels>: specify the field height
		# -X / --x-column <column-name>: the FITS column containing the X coordinate of
		# the sources
		# -Y / --y-column <column-name>: the FITS column containing the Y coordinate of
		# the sources
		}

	# UGLY
	if sub.parity == 0:
		axyargs['--parity'] = 'pos'
	elif sub.parity == 1:
		axyargs['--parity'] = 'neg'

	cmd = 'augment-xylist ' + ' '.join(k + ((v and ' ' + str(v)) or '') for (k,v) in axyargs.items())
	logmsg('running: ' + cmd)
	(rtn, out, err) = run_command(cmd)
	if rtn:
		logmsg('out: ' + out)
		logmsg('err: ' + err)
		#bailout(job, 'Creating axy file failed: ' + err)
		return False

	logmsg('created axy file ' + axyfn)

	# create a temp dir and cd into it...
	# shell into compute server...
	logfn = 'log'
	cmd = ('(echo %(jobid)s; '
		   ' tar cf - --ignore-failed-read %(axyfile)s) | '
		   'ssh -x -T %(sshconfig)s 2>>%(logfile)s | '
		   'tar xf - --atime-preserve -m --exclude=%(axyfile)s '
		   '>>%(logfile)s 2>&1' %
		   dict(jobid='job-%i' % (job.id), axyfile=axyfn,
				sshconfig=sshconfig, logfile=logfn)
		   )
	#+ '; chmod 664 *; chgrp www-data *')

	print 'command:', cmd

	w = os.system(cmd)

	if not os.WIFEXITED(w):
		print 'Solver failed'
		return

	rtn = os.WEXITSTATUS(w)
	if rtn:
		logmsg('Solver failed with return value %i' % rtn)
		#bailout(job, 'Solver failed.')
		return

	logmsg('Solver completed successfully.')

	
	# Solved?



def main():
	nthreads = 1

	pool = None
	if nthreads > 1:
		pool = multiprocessing.Pool(nthreads)

	# multiprocessing.Lock for django db?

	while True:
		print 'Checking for new Submissions'
		# FIXME -- started, or finished?
		#newsubs = Submission.objects.all().filter(processing_started=False)
		print 'Found', Submission.objects.count(), 'submissions'
		for s in Submission.objects.all():
			print s
		#newsubs = Submission.objects.all().filter(processing_started__isnull=True)
		newsubs = Submission.objects.all()
		print 'Found', newsubs.count(), 'unstarted submissions'
		if newsubs.count() == 0:
			time.sleep(3)
			continue
		# FIXME -- order by user, etc
		for sub in newsubs:
			if pool:
				pool.apply_async(dosub, (sub,))
			else:
				dosub(sub)

	

if __name__ == '__main__':
	main()

