from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response, get_object_or_404
from django.core.context_processors import csrf
from django.template import Context, loader, RequestContext
from django.core.urlresolvers import reverse
from django.core.exceptions import ObjectDoesNotExist

#from django.contrib.auth import authenticate, login, get_user
import django.contrib.auth as auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from django import forms

from openid.consumer.consumer import Consumer, SUCCESS, CANCEL, FAILURE, SETUP_NEEDED
from openid.store.filestore import FileOpenIDStore

from astrometry.net2 import settings

from urllib import urlencode
#import urllib2
import re

# python manage.py runserver oven.cosmo.fas.nyu.edu:8000

ftpurl_re = re.compile(
	r'^ftp://'
	r'(?:(?:[A-Z0-9-]+\.)+[A-Z]{2,6}|' #domain...
	r'localhost|' #localhost...
	r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
	r'(?::\d+)?' # optional port
	r'(?:/?|/\S+)$', re.IGNORECASE)


class LoginForm(forms.Form):
	openid = forms.CharField()

	provider = forms.ChoiceField(choices=(
			('google', 'Google Account'),
			),
				     initial='google',
				     widget=forms.Select(
			attrs={#'id':'provider',
				'onchange':'providerChanged()',}
			))

def provideraccount_to_openid(provider, username):
	if provider == 'google':
		return 'http://www.google.com/profiles/' + username
	return username

class FtpOrHttpURLField(forms.URLField):
	def clean(self, value):
		#log('FtpOrHttpURLField.clean(', value, ')')
		try:
			val = super(FtpOrHttpURLField, self).clean(value)
			return val
		except ValidationError:
			pass
		if ftpurl_re.match(value):
			return value
		raise ValidationError(self.error_messages['invalid'])

class ForgivingURLField(FtpOrHttpURLField):
	def clean(self, value):
		#log('ForgivingURLField.clean(', value, ')')
		if value is not None and \
			   (value.startswith('http://http://') or
				value.startswith('http://ftp://')):
			value = value[7:]
		return super(ForgivingURLField, self).clean(value)

class SimpleURLForm(forms.Form):
	url = ForgivingURLField(initial='http://',
				widget=forms.TextInput(attrs={'size':'50'}))


#   1. The user enters their OpenID into a field on the consumer's site, and hits a login button.
#   2. The consumer site discovers the user's OpenID provider using the Yadis protocol.
#   3. The consumer site sends the browser a redirect to the OpenID provider. This is the authentication request as described in the OpenID specification.
#   4. The OpenID provider's site sends the browser a redirect back to the consumer site. This redirect contains the provider's response to the authentication request.


# Thanks to python-openid-django
def normalDict(request_data):
	"""
	Converts a django request MutliValueDict (e.g., request.GET,
	request.POST) into a standard python dict whose values are the
	first value from each of the MultiValueDict's value lists.  This
	avoids the OpenID library's refusal to deal with dicts whose
	values are lists, because in OpenID, each key in the query arg set
	can have at most one value.
	"""
	d = {}
	for k,v in request_data.iteritems():
		print 'key', k, 'has val type', type(v)
		if type(v) is str or type(v) is unicode:
			d[k] = v
		else:
			d[k] = v[0]
	return d

@login_required
def submit_url(req):
	urlerr = None
	if len(req.POST):
		form = SimpleURLForm(req.POST)
		if form.is_valid():
			url = form.cleaned_data['url']
			print 'Submit url', url
			#submission = Submission(user = req.user,
			#				filetype = 'image',
			#				datasrc = 'url',
			#				url = url)
			#submission.save()
			#submit_submission(req, submission)
			#return HttpResponseRedirect(get_status_url(submission.subid))
		else:
			urlerr = form['url'].errors[0]
	else:
		#if 'jobid' in req.session:
		#	del req.session['jobid']
		form = SimpleURLForm()

	print 'user:', req.user
	print 'authenticated:', req.user.is_authenticated()

	c = RequestContext(req, {
			'form' : form,
			'urlerr' : urlerr,
			'actionurl': reverse(submit_url),
			})
	c.update(csrf(req))
	return render_to_response('submiturl.html', c)


@login_required
def submit_file(req):
	return HttpResponse('pass')

@login_required
def submit_full(req):
	return HttpResponse('pass')

def login_openid_done(req):
	print 'openid done.'

	# URL will be like: http://oven.cosmo.fas.nyu.edu:8000/login-openid?janrain_nonce=2010-06-27T21%3A11%3A05ZGUVm0I&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.mode=id_res&openid.op_endpoint=https%3A%2F%2Fwww.google.com%2Faccounts%2Fo8%2Fud%3Fsource%3Dprofiles&openid.response_nonce=2010-06-27T21%3A11%3A06ZqMy72nnKMzOzWw&openid.return_to=http%3A%2F%2Foven.cosmo.fas.nyu.edu%3A8000%2Flogin-openid%3Fjanrain_nonce%3D2010-06-27T21%253A11%253A05ZGUVm0I&openid.assoc_handle=AOQobUcWsAqu3NZaW4XwauYNP2E9XdJG11_fO5C0LEMNweCAE1ShbzSB&openid.signed=op_endpoint%2Cclaimed_id%2Cidentity%2Creturn_to%2Cresponse_nonce%2Cassoc_handle&openid.sig=y2gaOuQL3bsyLSZH0xoXow%2BOzco%3D&openid.identity=http%3A%2F%2Fwww.google.com%2Fprofiles%2Fdstndstn&openid.claimed_id=http%3A%2F%2Fwww.google.com%2Fprofiles%2Fdstndstn

	con = Consumer(req.session, FileOpenIDStore(settings.openid_dir))
	myurl = req.build_absolute_uri()
	print 'my url:', myurl

	# OpenID 2 can send arguments as either POST body or GET query
	# parameters.
	args = normalDict(req.GET)
	if req.method == 'POST':
		args.update(normalDict(req.POST))

	resp = con.complete(args, myurl)
	#resp.status = #SUCCESS, CANCEL, FAILURE, or SETUP_NEEDED.

	print 'response type:', type(resp)
	print 'response:', resp
	# add resp.message to message queue and redirect to login page...
	print 'display identifier:', resp.getDisplayIdentifier()

	if resp.status == SUCCESS:
		print 'claimed id:', resp.identity_url
		#user = authenticate(username=resp.identity_url, password='')
		# user = authenticate(user)
		#if user 
		# login(req, user)
		#user = auth.get_user(req) #resp.identity_url)
		try:
			user = User.objects.get(username=resp.identity_url)
			print 'Got user:', user
			print 'password:', user.password
			user = auth.authenticate(username=resp.identity_url, password='password')
			print 'After authenticate: user:', user
			auth.login(req, user)
			print 'user logged in.'
			# redirect to dashboard...
			return HttpResponseRedirect(reverse(submit_url))
		except ObjectDoesNotExist:
			print 'User not found'
			# Create user?


	return HttpResponse('<html>welcome back from openid land, ' + req.session.get('openid') + ' <a href="/login">Login</a></html>')

# Port-forwarding:
#   ssh -R "*:9000:localhost:8000" astrometry

urlproxy = {'localhost:8000': 'astrometry.net:9000'}


def login(req):
	if req.method == 'POST':
		form = LoginForm(req.POST)
		if form.is_valid():
			oid = form.cleaned_data.get('openid', None)
			prov = form.cleaned_data.get('provider', None)
			oid = provideraccount_to_openid(prov, oid)
			#return HttpResponseRedirect('/thanks/') # Redirect after POST
			req.session['openid'] = oid
			print 'openid:', oid
			con = Consumer(req.session, FileOpenIDStore(settings.openid_dir))
			authreq = con.begin(oid)

			#myurl = settings.openid_realm
			myurl = 'http://' + req.get_host()
			rtnurl = 'http://' + req.get_host() + reverse(login_openid_done)
			print 'my url:', myurl
			print 'return url:', rtnurl
			for k,v in urlproxy.items():
				rtnurl = rtnurl.replace(k, v)
			print 'return url:', rtnurl

			if authreq.shouldSendRedirect():
				url = authreq.redirectURL(realm=myurl, return_to=rtnurl)
				print 'redirecting to', url
				return HttpResponseRedirect(url)

			form_html = authreq.htmlMarkup(
				myurl, rtnurl,
				form_tag_attrs={'id':'openid_message'},
				immediate=False)
			return HttpResponse(form_html)
			
	else:
		form = LoginForm()

	c = {'form': form,
		 'next': '/home'}
	c.update(csrf(req))
	return render_to_response('login.html', c)

def logout(req):
	return HttpResponse('not implemented')
