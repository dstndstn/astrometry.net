import django.contrib.auth as auth

from django import forms as forms
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.forms import ValidationError, ModelForm
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render_to_response
from django.core.mail import send_mail
from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.template import Context, RequestContext, loader

from astrometry.net.portal.models import UserProfile
from astrometry.net.portal.log import log as logmsg

class NonDuplicateEmailField(forms.EmailField):
	def clean(self, value):
		logmsg('cleaning non-duplicate email field:', value)
		value = value.lower()
		try:
			u = User.objects.get(email=value)
			logmsg('found user', u.username)
		except ObjectDoesNotExist:
			# good, this email address doesn't exist yet
			return value

		# re-requesting an activation key is allowed.
		if not UserProfile.for_user(u).activated:
			return value

		raise ValidationError('That email address is already registered.')


class NewAccountForm(forms.Form):
	name = forms.CharField(required=False)
	email = NonDuplicateEmailField()

def activateaccount(request):
	key = request.GET.get('key')
	if key is None:
		return HttpResponseBadRequest('missing key')
	try:
		profile = UserProfile.objects.get(activation_key=key)
	except ObjectDoesNotExist:
		return HttpResponseBadRequest('no such key, or this key has already been used to activate this account')

	user = profile.user
	# log the user in and redirect to password-changing page.
	pw = 'temp'+key
	user.set_password(pw)
	user.save()
	user = auth.authenticate(username=user.username, password=pw)
	auth.login(request, user)
	user.set_unusable_password()
	user.save()
	request.session['allow_set_password'] = True
	return HttpResponseRedirect(reverse('astrometry.net.setpassword'))

def setpassword(request):
	if not request.session.get('allow_set_password', False):
		return HttpResponseRedirect(reverse('astrometry.net.changepassword'))
	if request.POST:
		form = auth.forms.SetPasswordForm(request.user, request.POST)
		if form.is_valid():
			request.user.set_password(form.cleaned_data['new_password1'])
			request.user.save()
			request.session['allow_set_password'] = False

			profile = UserProfile.for_user(request.user)
			profile.activated = True
			profile.activation_key = ''
			profile.save()

			return render_to_response(
				'portal/message.html',
				{ 'message': 'Password Set Successfully.  Use the menu above to start using the system.	 Enjoy!' },
				context_instance = RequestContext(request))
	else:
		form = auth.forms.SetPasswordForm(request.user)
	return render_to_response(
		'portal/setpassword.html',
		{ 'form': form },
		context_instance = RequestContext(request))


def newaccount(request):
	if len(request.POST):
		form = NewAccountForm(request.POST)
		if form.is_valid():
			email = form.cleaned_data['email']
			name = form.cleaned_data['name']

			# -create User account with inactive password
			if User.objects.filter(username=email).count():
				user = User.objects.get(username=email)
				user.set_unusable_password()
			else:
				user = User.objects.create_user(email, email, None)
			user.first_name = name
			user.save()

			# -generate an activation key and save it
			profile = UserProfile.for_user(user)
			key = profile.new_activation_key()

			# -send an email
			send_mail('Your Astrometry.net account',
					  ('In order to activate your Astrometry.net account, please visit the URL below.'
					   + '\n\n'
					   + '' + request.build_absolute_uri(reverse(activateaccount)) + '?key=' + key
					   + '\n\n'
					   + 'Thanks for trying Astrometry.net!'
					   ),
					  'Astrometry.net Accounts <alpha@astrometry.net>',
					  [email], fail_silently=False)

			# -tell the user what happened
			return render_to_response(
				'portal/message.html',
				{
				'message' : 'An email has been sent with instructions on how to activate your account.',
				},
				context_instance = RequestContext(request))
	else:
		form = NewAccountForm()

	return render_to_response(
		'portal/newaccount.html',
		{ 'form' : form },
		context_instance = RequestContext(request))

def logout(request):
	auth.logout(request)
	return HttpResponseRedirect(reverse('astrometry.net.portal.newjob.newurl'))

class UserProfileForm(ModelForm):
	class Meta:
		model = UserProfile

@login_required
def userprefs(request):
	prefs = UserProfile.for_user(request.user)
	if request.POST:
		form = UserProfileForm(request.POST)
	else:
		form = UserProfileForm({
			'exposejobs': prefs.exposejobs,
			})

	if request.POST and form.is_valid():
		prefs.set_expose_jobs(form.cleaned_data['exposejobs'])
		prefs.save()
		msg = 'Preferences Saved'
	else:
		msg = None
		
	ctxt = {
		'msg' : msg,
		'form' : form,
		}
	t = loader.get_template('portal/userprefs.html')
	c = RequestContext(request, ctxt)
	return HttpResponse(t.render(c))

