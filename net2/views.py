from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response, get_object_or_404
from django.core.context_processors import csrf
from django.template import Context, loader, RequestContext
from django.core.urlresolvers import reverse

from django import forms

from openid.consumer.consumer import Consumer
from openid.store.filestore import FileOpenIDStore

from astrometry.net2 import settings

from urllib import urlencode
#import urllib2

class LoginForm(forms.Form):
	openid = forms.CharField()


#   1.   The user enters their OpenID into a field on the consumer's site, and hits a login button.
#   2. The consumer site discovers the user's OpenID provider using the Yadis protocol.
#   3. The consumer site sends the browser a redirect to the OpenID provider. This is the authentication request as described in the OpenID specification.
#   4. The OpenID provider's site sends the browser a redirect back to the consumer site. This redirect contains the provider's response to the authentication request.


def login_openid_done(req):
	return HttpResponse('welcome back from openid land, ', req.session.get('openid'))

# Port-forwarding:
#   ssh -R "*:9000:localhost:8000" astrometry

urlproxy = {'localhost:8000': 'astrometry.net:9000'}


def login(req):
	if req.method == 'POST':
		form = LoginForm(req.POST)
		if form.is_valid():
			oid = form.cleaned_data.get('openid', None)
			#return HttpResponseRedirect('/thanks/') # Redirect after POST
			req.session['openid'] = oid
			print 'openid:', oid
			con = Consumer(req.session, FileOpenIDStore(settings.openid_dir))
			authreq = con.begin(oid)

			myurl = settings.openid_realm
			rtnurl = 'http://' + req.get_host() + reverse(login_openid_done)
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
