from django.contrib import auth
from django.contrib.auth import authenticate
from django.contrib.auth.models import User

from django.http import HttpResponse, HttpResponseBadRequest
from django.core.urlresolvers import reverse
from django.core.exceptions import ObjectDoesNotExist

from astrometry.net.portal.job import Job, Tag, Submission
from astrometry.net.portal.log import log as logmsg
from astrometry.net import settings
from astrometry.net.portal.api_util import *
from astrometry.net.portal.newjob import submit_submission

import os
import pickle
import base64
from urlparse import urlparse
from Crypto.PublicKey import RSA

json_type = 'text/plain' # 'application/json'

#class HttpResponseErrorJson(HttpResponseBadRequest):
class HttpResponseErrorJson(HttpResponse):
	def __init__(self, errstring):
		args = { 'status': 'error',
				 'errormessage': errstring }
		doc = python2json(args)
		super(HttpResponseErrorJson, self).__init__(doc, content_type=json_type)

class HttpResponseJson(HttpResponse):
	def __init__(self, args):
		doc = python2json(args)
		super(HttpResponseJson, self).__init__(doc, content_type=json_type)

def create_session(key):
	from django.conf import settings
	engine = __import__(settings.SESSION_ENGINE, {}, {}, [''])
	sess = engine.SessionStore(key)
	# sess.session_key creates a new key if necessary.
	return sess

# decorator for extracting JSON arguments from a POST.
def requires_json_args(handler):
	def handle_request(request, *pargs, **kwargs):
		#print 'requires_json_args decorator running.'
		json = request.POST.get('request-json')
		if not json:
			return HttpResponseErrorJson('no json')
		args = json2python(json)
		if args is None:
			return HttpResponseErrorJson('failed to parse JSON: ', json)
		request.json = args
		#print 'json:', request.json
		return handler(request, *pargs, **kwargs)
	return handle_request

# decorator for retrieving the user's session based on a session key in JSON.
# requires "request.json" to exist: you probably need to precede this decorator
# by the "requires_json_args" decorator.
def requires_json_session(handler):
	def handle_request(request, *args, **kwargs):
		#print 'requires_json_session decorator running.'
		if not 'session' in request.json:
			return HttpResponseErrorJson('no "session" in JSON.')
		key = request.json['session']
		session = create_session(key)
		if not session.exists(key):
			return HttpResponseErrorJson('no session with key "%s"' % key)
		request.session = session
		#print 'session:', request.session
		resp = handler(request, *args, **kwargs)
		session.save()
		# remove the session from the request so that SessionMiddleware
		# doesn't try to set cookies.
		del request.session
		return resp
	return handle_request

def requires_json_login(handler):
	def handle_request(request, *args, **kwargs):
		#print 'requires_json_login decorator running.'
		user = auth.get_user(request)
		#print 'user:', request.session
		if not user.is_authenticated():
			return HttpResponseErrorJson('user is not authenticated.')
		return handler(request, *args, **kwargs)
	return handle_request
	
@requires_json_args
def login(request):
	uname = request.json.get('username')
	password = request.json.get('password')
	if uname is None or password is None:
		return HttpResponseErrorJson('need "username" and "password".')

	user = authenticate(username=uname, password=password)
	if user is None:
		return HttpResponseErrorJson('incorrect username/password')

	# User has successfully logged in.	Create session.
	session = create_session(None)

	request.session = session
	auth.login(request, user)
	del request.session

	if False:
		# generate pubkey
		keybits = 512
		privkey = RSA.generate(keybits, os.urandom)
		privkeystr = base64.encodestring(pickle.dumps(privkey.__getstate__()))
		pubkey = privkey.publickey()
		pubkeystr = base64.encodestring(pickle.dumps(pubkey.__getstate__()))

		# client encodes like this:
		message = 'Mary had a little hash'
		ckey = RSA.RSAobj()
		ckey.__setstate__(pickle.loads(base64.decodestring(pubkeystr)))
		secret = ckey.encrypt(message, K='')
		logmsg('secret = ', secret)

		# server decodes like this:
		skey = RSA.RSAobj()
		skey.__setstate__(pickle.loads(base64.decodestring(privkeystr)))
		dmessage = skey.decrypt(secret)
		logmsg('decrypted = ', dmessage)

		session['private_key'] = privkeystr
		session['public_key'] = pubkeystr

	session.save()

	key = session.session_key
	return HttpResponseJson({ 'status': 'success',
							  'message': 'authenticated user: ' + str(user.email),
							  'session': key,
							  #'pubkey': pubkeystr,
							  })

@requires_json_args
def amiloggedin(request):
	if not 'session' in request.json:
		return HttpResponseJson({ 'loggedin': False,
								  'reason': 'no session in args'})
	key = request.json['session']
	session = create_session(key)
	if not session.exists(key):
		return HttpResponseJson({ 'loggedin': False,
								  'reason': 'no such session with key: '+key})
	request.session = session
	user = auth.get_user(request)
	del request.session

	if not user.is_authenticated():
		return HttpResponseJson({ 'loggedin': False,
								  'reason': 'user is not authenticated.'})
	return HttpResponseJson({ 'loggedin': True,
							  'username': user.username})


@requires_json_args
@requires_json_session
@requires_json_login
def logout(request):
	uname = request.user.username
	auth.logout(request)
	return HttpResponseJson({ 'status': 'Success',
							  'message': 'User "%s" logged out.' % uname })

@requires_json_args
@requires_json_session
@requires_json_login
def jobstatus(request):
	if not 'jobid' in request.json:
		return HttpResponseErrorJson('no "jobid" in request args')
	jobid = request.json['jobid']
	try:
		job = Job.objects.get(jobid=jobid)
	except ObjectDoesNotExist:
		return HttpResponseErrorJson('no job with id "%s"' % jobid)
	if not job.can_be_viewed_by(request.user):
		return HttpResponseErrorJson('permission denied')
	return HttpResponseJson({ 'jobid': jobid,
							  'status': job.format_status_full(),
							  'user_tags': [ t.text for t in job.get_user_tags() ],
							  'machine_tags': [ t.text for t in job.get_machine_tags() ],
							  })


@requires_json_args
@requires_json_session
@requires_json_login
def substatus(request):
	if not 'subid' in request.json:
		return HttpResponseErrorJson('no "subid" in request args')
	subid = request.json['subid']
	try:
		sub = Submission.objects.get(subid=subid)
	except ObjectDoesNotExist:
		return HttpResponseErrorJson('no submission with id "%s"' % subid)
	return HttpResponseJson({ 'subid': subid,
							  'status': sub.format_status(),
							  'jobs-queued':   [j.jobid for j in sub.jobs_queued()],
							  'jobs-running':  [j.jobid for j in sub.jobs_running()],
							  'jobs-solved':   [j.jobid for j in sub.jobs_solved()],
							  'jobs-unsolved': [j.jobid for j in sub.jobs_unsolved()],
							  'jobs-error':    [j.jobid for j in sub.jobs_error()],
							  })

@requires_json_args
@requires_json_session
@requires_json_login
def submit_url(request):
	url = request.json.get('url')
	if not url:
		return HttpResponseErrorJson('no "url" in request args')

	p = urlparse(url)

	allowedschemes = ['http', 'ftp']
	if not p.scheme in allowedschemes:
		return HttpResponseErrorJson('url scheme "%s" not allowed (only http, ftp).' % p.scheme)

	submission = Submission(user=request.user,
							filetype='image',
							datasrc='url',
							url=url)
	submission.save()
	logmsg("submitted submission:", submission)
	submit_submission(request, submission)
	return HttpResponseJson({ 'subid': submission.subid })
