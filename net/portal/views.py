import datetime
import logging
import math
import os
import os.path
import random
import sha
import tempfile
import time

from urllib import urlencode

from django import forms as forms

import django.contrib.auth as auth
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

from django.db import models
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.forms import widgets, ValidationError
from django.forms import ModelForm
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render_to_response
from django.core.mail import send_mail

import astrometry.net.portal.mercator as merc
from astrometry.net.portal.models import UserProfile
from astrometry.net.portal.job import Job, Submission, DiskFile, Tag
from astrometry.net.portal.convert import convert, get_objs_in_field
from astrometry.net.portal.log import log
from astrometry.net.portal import tags
from astrometry.net.portal import nearby
from astrometry.util.file import file_size, read_file, write_file
from astrometry.net import settings
from astrometry.util import sip

def urlescape(s):
	return s.replace('&', '&amp;')

def get_status_url(jobid):
	return reverse(jobstatus, args=[jobid])

def get_file_url(job, fn, args=None):
	url = reverse(getfile, args=[job.jobid, fn])
	if args:
		url += '?' + args
	return url

def get_job(jobid):
	if jobid is None:
		return None
	jobs = Job.objects.all().filter(jobid=jobid)
	if len(jobs) != 1:
		#log('Found %i jobs, not 1' % len(jobs))
		return None
	job = jobs[0]
	return job

def get_job_and_sub(request, args, kwargs):
	jobid = None
	if 'jobid' in kwargs:
		jobid = kwargs['jobid']
	elif request.GET:
		jobid = request.GET.get('jobid')
	elif request.POST:
		jobid = request.POST.get('jobid')
	job = None
	#log('jobid', jobid)
	if jobid:
		job = get_job(jobid)
	sub = None
	subid = request.REQUEST.get('subid')
	if subid:
		sub = get_submission(subid)
	if sub is None and job is None and jobid is not None:
		sub = get_submission(jobid)
	return (job, sub)

# a decorator for calls that need a valid jobid; puts the job in
# request.job
def needs_job(handler):
	def handle_request(request, *args, **kwargs):
		(job,sub) = get_job_and_sub(request, args, kwargs)
		if not job:
			return HttpResponse('no job with jobid ' + jobid)
		request.job = job
		return handler(request, *args, **kwargs)
	return handle_request

def needs_sub(handler):
	def handle_request(request, *args, **kwargs):
		(job,sub) = get_job_and_sub(request, args, kwargs)
		if not sub:
			return HttpResponse('need subid')
		request.sub = sub
		return handler(request, *args, **kwargs)
	return handle_request

# a decorator for calls that need a valid jobid; puts the job in
# request.job
def wants_job_or_sub(handler):
	def handle_request(request, *args, **kwargs):
		(job,sub) = get_job_and_sub(request, args, kwargs)
		request.job = job
		request.sub = sub
		return handler(request, *args, **kwargs)
	return handle_request

def get_submission(subid):
	if subid is None:
		return None
	subs = Submission.objects.all().filter(subid=subid)
	if len(subs) != 1:
		return None
	return subs[0]

def getsessionjob(request):
	if not 'jobid' in request.session:
		log('no jobid in session')
		return None
	jobid = request.session['jobid']
	return get_job(jobid)

# looks for a user=xxx in the GET.
def getuser(request):
	uname = request.GET.get('user')
	if not uname:
		return None
	try:
		user = User.objects.get(username=uname)
	except ObjectDoesNotExist:
		return None
	return user

@login_required
@needs_sub
def sub_add_tag(request):
	tag = request.REQUEST.get('tag', None)
	if not tag:
		return HttpResponse('no tag')

	tagall = request.REQUEST.get('tagall')
	tagsolved = request.REQUEST.get('tagsolved')

	now = Job.timenow()
	nadded = 0

	jobs = request.sub.jobs.all()
	if tagsolved:
		jobs = jobs.filter(status='Solved')

	for job in jobs:
		if not job.can_add_tag(request.user):
			return HttpResponse('not permitted')
		t = Tag(job=job,
				user=request.user,
				machineTag=False,
				text=tag,
				addedtime=now)
		if not t.is_duplicate():
			t.save()
			nadded += 1

	request.user.message_set.create(message='Tag <i>%s</i> added to %i jobs' % (tag, nadded))

	redir = request.REQUEST.get('redirect', None)
	if redir:
		return HttpResponseRedirect(redir)
	return HttpResponse('Tag added')

@login_required
@wants_job_or_sub
def joblist(request):
	myargs = QueryDict('', mutable=True)
	job = request.job
	sub = request.sub
	n = int(request.GET.get('n', '50'))
	start = int(request.GET.get('start', '0'))
	if n:
		end = start + n
	else:
		end = -1

	# are we redirecting to gmaps?
	togmaps = 'gmaps' in request.GET

	format = request.GET.get('format', 'html')

	kind = request.GET.get('type')
	if not kind in [ 'user', 'user-multi', 'nearby', 'tag', 'sub' ]:
		kind = 'user'

	if sub:
		myargs['subid'] = sub.subid
	if job:
		myargs['jobid'] = job.jobid
	if start:
		myargs['start'] = start
	if n:
		myargs['n'] = n
	if format != 'html':
		myargs['format'] = format
	myargs['type'] = kind

	colnames = {
		'mark': 'Selected',
		'jobid' : 'Job Id',
		'status' : 'Status',
		'starttime' : 'Start Time',
		'finishtime' : 'Finish Time',
		'enqueuetime' : 'Enqueued Time',
		'radec' : '(RA, Dec)',
		'fieldsize' : 'Field size',
		'tags' : 'Tags',
		'desc' : 'Description',
		'objsin' : 'Objects',
		'thumbnail' : 'Thumbnail',
		'annthumb' : 'Annotated Thumbnail',
		'user' : 'User',
		# DEBUG
		'diskfile' : 'Diskfile',
		'subid': 'Submission',
		'submittime': 'Submit Time',
		'njobs': 'Number of Jobs',
		}

	allcols = colnames.keys()

	cols = ','.join(request.GET.getlist('cols'))
	if cols:
		cols = cols.split(',')
		okcols = []
		for c in cols:
			if c in allcols:
				okcols.append(c)
		cols = okcols

	ctxt = {}

	ajaxupdate = False
	title = None
	# allow duplicate DiskFiles?
	duplicates = False
	reload_time = None
	links = []
	subs = []
	jobs = []

	if kind in ['user', 'user-multi']:
		user = getuser(request)
		if not user:
			user = request.user
		myargs['user'] = user.username
		# find multi-job submissions from this user.
		multisubs = Submission.objects.all().filter(user=user, multijob=True)

	if kind == 'user':
		jobs = Job.objects.all().filter(submission__user=user).order_by('-enqueuetime', '-starttime')
		if user != request.user:
			jobs = jobs.filter(exposejob=True)

		if multisubs.count():
			url = reverse(joblist) + '?' + urlencode({'type':'user-multi', 'user':user.username})
			links.append(('Multi-job submissions by this user', url))
		
		N = jobs.count()
		if not cols:
			cols = [ 'thumbnail', 'jobid', 'status' ]
		title = 'Jobs submitted by <i>%s</i>' % user.username

	elif kind == 'user-multi':
		multisubs = multisubs.order_by('-submittime')
		N = multisubs.count()
		if not cols:
			cols = [ 'subid', 'submittime', 'status', 'njobs' ]
		title = 'Multi-job submissions from <i>%s</i>' % user.username

	elif kind == 'nearby':
		if job is None:
			return HttpResponse('no job')
		tags = nearby.get_tags_nearby(job)
		if tags is None:
			return HttpResponse('error getting nearby tags')
		N = tags.count()
		if not cols:
			cols = [ 'thumbnail', 'jobid', 'user' ]
			title = 'Jobs near <i>%s</i>' % job.jobid

	elif kind == 'tag':
		tagtxt = request.GET.get('tagtext')
		if not tagtxt:
			return HttpResponse('no tag')
		myargs['tagtext'] = tagtxt
		tags = Tag.objects.all().filter(text=tagtxt)
		if tags.count() == 0:
			return HttpResponse('no such tag')

		tags = tags.filter(job__duplicate=False)
		N = tags.count()

		if not cols:
			cols = [ 'thumbnail', 'jobid', 'user' ]
		title = 'Jobs tagged with <i>%s</i>' % tagtxt

	elif kind == 'sub':
		if sub is None:
			return HttpResponse('no sub')
		job = None
		jobs = sub.jobs.all().order_by('enqueuetime', 'starttime', 'jobid')
		N = jobs.count()
		if 'onejobjump' in request.GET and N == 1 and sub.alljobsadded:
			return HttpResponseRedirect(reverse(jobstatus, args=[jobs[0].jobid]))

		if not cols:
			cols = [ 'mark', 'jobid', 'status', 'starttime', 'finishtime' ]
		if sub.status == 'Queued' or sub.status == 'Running':
			if N == 1:
				reload_time = 2
			else:
				reload_time = 15
		ajaxupdate = True

		title = 'Jobs belonging to submission <i>' + sub.subid + '</i>'
		subtitle = 'Submission <i>' + sub.subid + '</i>'


	if togmaps:
		from astrometry.net.tile.models import *

		if kind == 'tag':
			jobs = [t.job for t in tags]

		jobset = MapImageSet()
		jobset.save()
		for job in jobs:
			calib = job.calibration
			if not calib:
				continue

			try:
				img = MapImage.objects.get(job=job)
			except ObjectDoesNotExist:
				img = MapImage(job=job,
							   ramin=calib.ramin,
							   ramax=calib.ramax,
							   decmin=calib.decmin,
							   decmax=calib.decmax)
				img.save()
			jobset.images.add(img)

		url = (reverse('astrometry.net.tile.views.index') +
			   ('?imageset=%i' % jobset.id) +
			   '&layers=tycho,userimages&arcsinh')

		return HttpResponseRedirect(url)

	myargs['cols'] = ','.join(cols)

	if end > 0 and end < N:
		args = myargs.copy()
		args['start'] = end
		ctxt['nexturl'] = request.path + '?' + args.urlencode()
		args['start'] = start + n * ((N - start) / n)
		ctxt['lasturl'] = request.path + '?' + args.urlencode()
	if start > 0:
		args = myargs.copy()
		prev = max(0, start-n)
		args['start'] = prev
		ctxt['prevurl'] = request.path + '?' + args.urlencode()
		if prev != 0:
			args['start'] = 0
			ctxt['firsturl'] = request.path + '?' + args.urlencode()
		
	ctxt['firstnum'] = max(0, start) + 1
	ctxt['lastnum']	 = end == -1 and N or min(N, end)
	ctxt['totalnum']  = N

	addcols = [c for c in allcols if c not in cols]

	if kind == 'user':
		jobs = jobs[start:end]

	elif kind == 'user-multi':
		multisubs = multisubs[start:end]
		subs = multisubs
		addcols = []

	elif kind == 'nearby':
		tags = tags[start:end]
		jobs = [t.job for t in tags]

	elif kind == 'tag':
		tags = tags.order_by('addedtime')
		tags = tags[start:end]
		jobs = [t.job for t in tags]

	elif kind == 'sub':
		jobs = jobs[start:end]
		if N == 0:
			title = subtitle

	# "rjobs": rendered jobs.
	rjobs = []
	for i, job in enumerate(jobs):
		rend = []
		jobn = start + i
		for c in cols:
			t = ''
			tdclass = 'c'
			if c == 'jobid':
				t = ('<a href="'
					 + urlescape(get_status_url(job.jobid))
					 + '">'
					 + job.jobid
					 + '</a>')
			elif c == 'mark':
				t = ('<input type="checkbox" name="mark-job-%s" id="mark-%s" />' %
					 (job.jobid, job.jobid))
			elif c == 'starttime':
				t = job.format_starttime_brief()
			elif c == 'finishtime':
				t = job.format_finishtime_brief()
			elif c == 'enqueuetime':
				t = job.format_enqueuetime_brief()
			elif c == 'status':
				t = job.format_status()
			elif c == 'radec':
				if job.solved():
					wcs = job.calibration.raw_tan
					(ra,dec) = wcs.get_field_center()
					t = '(%.2f, %.2f)' % (ra, dec)
			elif c == 'fieldsize':
				if job.solved():
					wcs = job.calibration.raw_tan
					(w, h, units) = wcs.get_field_size()
					t = '%.2f x %.2f %s' % (w, h, units)
			elif c == 'tags':
				tags = job.tags.all().filter(machineTag=False).order_by('addedtime')
				t = ', '.join([tag.text for tag in tags])
				log('tags:', tags)
				log('t:', t)
			elif c == 'desc':
				t = job.description or ''
			elif c == 'objsin':
				if job.solved():
					objs = get_objs_in_field(job)
					t = ', '.join([
						('<a href="' + request.path + '?'
						 + urlescape(urlencode({'type': 'tag',
												'tagtext': obj.encode('utf_8') }))
						 + '">' + obj + '</a>'
						 ) for obj in objs])
			elif c == 'thumbnail':
				t = ('<img src="'
					 + get_file_url(job, 'thumbnail')
					 + '" alt="Thumbnail" />')
			elif c == 'annthumb':
				if job.solved():
					t = ('<img src="'
						 + get_file_url(job, 'annotation-thumb')
						 + '" alt="Thumbnail" />')
			elif c == 'user':
				t = ('<a href="'
					 + reverse(joblist) + urlescape('?type=user&user='
					 + job.get_user().username)
					 + '">'
					 + job.get_user().username
					 + '</a>')
			# DEBUG
			elif c == 'diskfile':
				t = job.diskfile.filehash

			rend.append((tdclass, c, t))
		rjobs.append((rend, job.jobid, jobn))

	rsubs = []
	for i, sub in enumerate(subs):
		# this row.
		rend = []
		subn = start + i
		for c in cols:
			t = ''
			tdclass = 'c'
			if c == 'subid':
				t = ('<a href="'
					 + reverse(joblist) + urlescape('?type=sub&subid=' + sub.subid)
					 + '">'
					 + sub.subid
					 + '</a>')
			elif c == 'submittime':
				t = sub.format_submittime_brief()
			elif c == 'status':
				t = sub.status
			elif c == 'njobs':
				t = sub.jobs.count()
			rend.append((tdclass, c, t))
		rsubs.append((rend, sub.subid, subn))


	if format == 'xml':
		res = HttpResponse()
		res['Content-type'] = 'text/xml'

		res.write('<submission subid="%s">\n' % subid)
		if kind == 'sub' and sub.is_finished():
			res.write('	 <stop />\n')
		for (rend, jobid, jobn) in rjobs:
			res.write('	 <job jobid="%s" n="%i">\n' % (jobid, jobn))
			for (tdclass, c, t) in rend:
				res.write('	  <%s>%s</%s>\n' % (c, t, c))
			res.write('	 </job>\n')
		res.write('</submission>\n')
		return res

	else:
		cnames = [colnames.get(c) for c in cols]

		columns = []
		for c,n in zip(cols, cnames):
			args = request.GET.copy()
			delcols = cols[:]
			delcols.remove(c)
			args['cols'] = ','.join(delcols)
			delurl = request.path + '?' + args.urlencode()
			columns.append((c, n, delurl))

		addcolumns = zip(addcols, [colnames[c] for c in addcols])

		thisurl = request.path + '?' + myargs.urlencode()

		ctxt.update({
			'kind': kind,
			'thisurl' : thisurl,
			'links': links,
			'addcolumns' : addcolumns,
			'columns' : columns,
			'submission' : sub,
			'jobs' : jobs,
			'rjobs' : rjobs,
			'subs': subs,
			'rsubs' : rsubs,
			'reload_time' : reload_time,
			'gmaps' : thisurl + '&gmaps',
			'title' : title,
			})
		if ajaxupdate:
			ctxt['xmlsummaryurl'] = request.get_full_path() + '&format=xml'

		t = loader.get_template('portal/joblist.html')
		c = RequestContext(request, ctxt)
		return HttpResponse(t.render(c))

def run_variant(request):
	return HttpResponse('Not implemented')

@login_required
@needs_job
def job_set_description(request):
	job = request.job
	if job.get_user() != request.user:
		return HttpResponse('not your job')
	if not 'desc' in request.POST:
		return HttpResponse('no desc')
	desc = request.POST['desc']
	job.description = desc
	job.save()
	return HttpResponseRedirect(get_status_url(jobid))

@wants_job_or_sub
def jobstatus(request, jobid=None):
	#log('jobstatus: jobid=', jobid)
	job = request.job
	sub = request.sub
	#log('job is', job)
	#log('sub is', sub)
	if sub:
		jobs = sub.jobs.all()
		log('submission has %i jobs.' % len(jobs))
		if len(jobs) == 1:
			job = jobs[0]
			jobid = job.jobid
			log('submission has one job:', jobid)
			return HttpResponseRedirect(get_status_url(jobid))
		else:
			args = QueryDict('', mutable=True)
			args['subid'] = sub.subid
			args['type'] = 'sub'
			args['onejobjump'] = None
			args['cols'] = 'jobid,status'
			return HttpResponseRedirect(reverse(joblist) + '?' + args.urlencode())

	if job is None:
		return HttpResponse('no such job')

	if not job.can_be_viewed_by(request.user):
		return HttpResponse('The owner of this job (' + job.get_user().username + ') has not granted public access.')

	jobowner = (job.get_user() == request.user)

	df = job.diskfile
	submission = job.submission
	#log('jobstatus: Job is: ' + str(job))

	otherxylists = []
	# (image url, link url)
	#otherxylists.append(('test-image-url', 'test-link-url'))
	for n in (1,2,3,4):
		fn = convert(job, 'xyls-exists?', { 'variant': n })
		if fn is None:
			break
		otherxylists.append((get_file_url(job, 'sources-small', 'variant=%i' % n),
							 reverse(run_variant) + '?jobid=%s&variant=%i' % (job.jobid, n)))
							 #get_status_url(job.jobid) + '&run-xyls&variant=%i' % n))

	taglist = []
	for tag in job.tags.all().order_by('addedtime'):
		if tag.machineTag:
			continue
		tag.canremove = tag.can_remove_tag(request.user)
		taglist.append(tag)

	ctxt = {
		'jobid' : job.jobid,
		'jobstatus' : job.format_status_full(),
		'jobsolved' : job.solved(),
		'jobsubmittime' : submission.format_submittime(),
		'jobstarttime' : job.format_starttime(),
		'jobfinishtime' : job.format_finishtime(),
		'logurl' : get_file_url(job, 'blind.log'),
		'job' : job,
		'submission' : job.submission,
		'joburl' : (submission.datasrc == 'url') and submission.url or None,
		'jobfile' : (submission.datasrc == 'file') and submission.uploaded.userfilename or None,
		'jobscale' : job.friendly_scale(),
		'jobparity' : job.friendly_parity(),
		'diskfileurl': get_file_url(job, 'origfile'),
		'needs_medium_scale' : job.diskfile.needs_medium_size(),
		'sources' : get_file_url(job, 'sources-medium'),
		'sources_big' : get_file_url(job, 'sources-big'),
		'sources_small' : get_file_url(job, 'sources-small'),
		'redgreen_medium' : get_file_url(job, 'redgreen'),
		'redgreen_big' : get_file_url(job, 'redgreen-big'),
		#'otherxylists' : otherxylists,
		'jobowner' : job.get_user() == request.user,
		'exposejob': job.is_exposed(),
		'tags' : taglist,
		'view_tagtxt_url' : reverse(joblist) + '?type=tag&tagtext=',
		'view_nearby_url' : reverse(joblist) + '?type=nearby&jobid=' + job.jobid,
		'view_user_url' : reverse(joblist) + '?type=user&user=',
		'set_description_url' : reverse(job_set_description),
		'add_tag_url' : reverse(tags.job_add_tag),
		'remove_tag_url' : reverse(tags.job_remove_tag) + '?',
		}

	if job.solved():
		wcsinfofn = convert(job, 'wcsinfo')
		f = open(wcsinfofn)
		wcsinfotxt = f.read()
		f.close()
		wcsinfo = {}
		for ln in wcsinfotxt.split('\n'):
			s = ln.split(' ')
			if len(s) == 2:
				wcsinfo[s[0]] = s[1]

		ctxt.update({'racenter' : '%.2f' % float(wcsinfo['ra_center']),
					 'deccenter': '%.2f' % float(wcsinfo['dec_center']),
					 'fieldw'	: '%.2f' % float(wcsinfo['fieldw']),
					 'fieldh'	: '%.2f' % float(wcsinfo['fieldh']),
					 'fieldunits': wcsinfo['fieldunits'],
					 'racenter_hms' : wcsinfo['ra_center_hms'],
					 'deccenter_dms' : wcsinfo['dec_center_dms'],
					 'orientation' : '%.3f' % float(wcsinfo['orientation']),
					 'pixscale' : '%.4g' % float(wcsinfo['pixscale']),
					 'parity' : (float(wcsinfo['det']) > 0 and 'Positive' or 'Negative'),
					 'wcsurl' : get_file_url(job, 'wcs.fits'),
					 'newfitsurl' : get_file_url(job, 'new.fits'),
					 'indexxyurl' : get_file_url(job, 'index.xy.fits'),
					 'indexrdurl' : get_file_url(job, 'index.rd.fits'),
					 'fieldxyurl' : get_file_url(job, 'field.xy.fits'),
					 'fieldrdurl' : get_file_url(job, 'field.rd.fits'),
					 })

		ctxt['objsinfield'] = get_objs_in_field(job)

		# deg
		fldsz = math.sqrt(df.imagew * df.imageh) * float(wcsinfo['pixscale']) / 3600.0

		url = (reverse('astrometry.net.tile.views.get_tile') +
			   '?layers=tycho,grid,userboundary' +
			   '&arcsinh&wcsfn=%s' % job.get_relative_filename('wcs.fits'))
		smallstyle = '&w=300&h=300&lw=3'
		largestyle = '&w=1024&h=1024&lw=5'
		steps = [ {				 'gain':0,	  'dashbox':0.1,   'center':False },
				  {'limit':18,	 'gain':-0.5, 'dashbox':0.01,  'center':True, 'dm':0.05	 },
				  {'limit':1.8,	 'gain':0.25,				   'center':True, 'dm':0.005 },
				  ]
		zlist = []
		for last_s in range(len(steps)):
			s = steps[last_s]
			if 'limit' in s and fldsz > s['limit']:
				log('break')
				break
		else:
			last_s = len(steps)

		for ind in range(last_s):
			s = steps[ind]
			if s['center']:
				xmerc = float(wcsinfo['ra_center_merc'])
				ymerc = float(wcsinfo['dec_center_merc'])
				ralo = merc.merc2ra(xmerc + s['dm'])
				rahi = merc.merc2ra(xmerc - s['dm'])
				declo = merc.merc2dec(ymerc - s['dm'])
				dechi = merc.merc2dec(ymerc + s['dm'])
				bb = [ralo, declo, rahi, dechi]
			else:
				bb = [0, -85, 360, 85]
			urlargs = ('&gain=%g' % s['gain']) + ('&bb=' + ','.join(map(str, bb)))
			if (ind < (last_s-1)) and ('dashbox' in s):
				urlargs += '&dashbox=%g' % s['dashbox']

			zlist.append([url + smallstyle + urlargs,
						  url + largestyle + urlargs])

		# HACK
		fn = convert(job, 'fullsizepng')
		url = (reverse('astrometry.net.tile.views.index') +
			   ('?zoom=%i&ra=%.3f&dec=%.3f&userimage=%s' %
				(int(wcsinfo['merczoom']), float(wcsinfo['ra_center']),
				 float(wcsinfo['dec_center']), job.get_relative_job_dir())))

		ctxt.update({
			'gmapslink' : url,
			'zoomimgs'	: zlist,
			'annotation': get_file_url(job, 'annotation'),
			'annotation_big' : get_file_url(job, 'annotation-big'),
			})

	else:
		logfn = job.get_filename('blind.log')
		if os.path.exists(logfn):
			f = open(logfn)
			logfiletxt = f.read()
			f.close()
			lines = logfiletxt.split('\n')
			lines = '\n'.join(lines[-16:])
			#log('job not solved')

			ctxt.update({
				'logfile_tail' : lines,
				})

	t = loader.get_template('portal/status.html')
	c = RequestContext(request, ctxt)
	return HttpResponse(t.render(c))

def getdf(idstr):
	dfs = DiskFile.objects.all().filter(filehash=idstr)
	if not len(dfs):
		return None
	df = dfs[0]
	return df

def send_file(filename, res=None, ctype='text/plain',
			  cdisposition='inline', cdownloadfn=None, clength=None):
	if res is None:
		res = HttpResponse()
	if clength is None:
		clength = file_size(filename)
	if cdownloadfn is not None:
		cdisposition = 'attachment; filename="' + cdownloadfn + '"'
	res['Content-Type'] = ctype
	res['Content-Disposition'] = cdisposition
	res['Content-Length'] = clength
	res.write(read_file(filename))
	return res

def getfield(request):
	if not 'fieldid' in request.GET:
		return HttpResponse('no fieldid')
	fieldid = request.GET['fieldid']
	df = getdf(fieldid)
	if not df:
		return HttpResponse('no such field')
		
	if not df.show():
		return HttpResponse('The owner of this field has not granted public access.')

	fn = df.get_path()
	ct = df.content_type() or 'application/octet-stream'

	return send_file(fn, ctype=ct)

@needs_job
def getfile(request, jobid=None, filename=None):
	job = request.job

	if not job.can_be_viewed_by(request.user):
		return HttpResponse('The owner of this job (' + job.get_user().username + ') has not granted public access.')

	variant = 0
	if 'variant' in request.GET:
		variant = int(request.GET['variant'])

	pngimages = [ 'annotation', 'annotation-big', 'annotation-thumb',
				  'sources-small', 'sources-medium', 'sources-big',
				  'redgreen', 'redgreen-big', 'thumbnail',
				  ]

	res = HttpResponse()
	t = datetime.datetime.utcnow() + datetime.timedelta(days=7)
	res['Expires'] = t.strftime('%a, %d %b %Y %H:%M:%S UTC')

	convertargs = {}
	if variant:
		convertargs['variant'] = variant

	### DEBUG - play with red-green colours.
	if filename.startswith('redgreen'):
		for x in ['red', 'green', 'rmarker', 'gmarker']:
			if x in request.GET:
				convertargs[x] = request.GET[x]

	if filename in pngimages:
		fn = convert(job, filename, convertargs)
		return send_file(fn, res=res, ctype='image/png')

	binaryfiles = [ 'wcs.fits', 'match.fits', 'field.xy.fits', 'field.rd.fits',
					'index.xy.fits', 'index.rd.fits', 'new.fits' ]
	if filename in binaryfiles:
		downloadfn = filename
		if filename == 'field.xy.fits':
			fn = job.get_axy_filename()
		elif filename in [ 'index.xy.fits', 'field.rd.fits', 'new.fits' ]:
			filename = convert(job, filename, convertargs)
			fn = job.get_filename(filename)
		else:
			fn = job.get_filename(filename)

		return send_file(fn, res=res, ctype='application/octet-stream',
						 cdownloadfn=downloadfn)

	textfiles = [ 'blind.log' ]
	if filename in textfiles:
		fn = job.get_filename(filename)
		return send_file(fn, res=res)

	if filename == 'origfile':
		#if not job.is_exposed():
		#	return HttpResponse('access to this file is forbidden.')
		df = job.diskfile
		ct = df.content_type() or 'application/octet-stream'
		fn = df.get_path()
		return send_file(fn, res=res, ctype=ct)

	return HttpResponse('bad f')

def printvals(request):
	if request.POST:
		log('POST values:')
		for k,v in request.POST.items():
			log('  %s = %s' % (str(k), str(v)))
	if request.GET:
		log('GET values:')
		for k,v in request.GET.items():
			log('  %s = %s' % (str(k), str(v)))
	if request.FILES:
		log('FILES values:')
		for k,v in request.FILES.items():
			log('  %s = %s' % (str(k), str(v)))

@login_required
@wants_job_or_sub
def changeperms(request):
	if not request.POST:
		return HttpResponse('no POST')
	prefs = UserProfile.for_user(request.user)

	log('changeperms:')
	for k,v in request.POST.items():
		log('  %s = %s' % (str(k), str(v)))
	if 'exposejob' in request.POST:
		expose = int(request.POST.get('exposejob', '-1'))
		if expose in [0, 1]:
			job = request.job
			if job is None:
				return HttpResponse('no job')
			if job.get_user() != request.user:
				return HttpResponse('not your job!')
			log('exposejob = %i' % expose)
			job.set_exposed(expose)
			job.save()
	if 'exposeall' in request.POST:
		exposeall = int(request.POST.get('exposeall', '-1'))
		if exposeall in [0, 1]:
			subs = Submission.objects.all().filter(user=request.user)
			for sub in subs:
				jobs = sub.jobs.all()
				for job in jobs:
					job.set_exposed(exposeall)
					job.save()

	if 'HTTP_REFERER' in request.META:
		return HttpResponseRedirect(request.META['HTTP_REFERER'])
	return HttpResponseRedirect(reverse(summary))

