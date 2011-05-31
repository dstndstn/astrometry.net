from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm, PasswordChangeForm
from django.contrib.auth.tokens import default_token_generator
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response, get_object_or_404
from django.contrib.sites.models import Site, RequestSite
from django.http import HttpResponseRedirect, Http404
from django.template import RequestContext
from django.utils.http import urlquote, base36_to_int
from django.utils.translation import ugettext as _
from django.contrib.auth.models import User
from django.views.decorators.cache import never_cache
from django.contrib.auth import authenticate, login

def password_reset_confirm(request, uidb36=None, token=None, template_name='registration/password_reset_confirm.html',
						   token_generator=default_token_generator, set_password_form=SetPasswordForm,
						   post_reset_redirect=None):
	"""
	View that checks the hash in a password reset link and presents a
	form for entering a new password.
	"""
	assert uidb36 is not None and token is not None # checked by URLconf
	if post_reset_redirect is None:
		post_reset_redirect = reverse('django.contrib.auth.views.password_reset_complete')
	try:
		uid_int = base36_to_int(uidb36)
	except ValueError:
		raise Http404

	user = get_object_or_404(User, id=uid_int)
	context_instance = RequestContext(request)

	if token_generator.check_token(user, token):
		context_instance['validlink'] = True
		if request.method == 'POST':
			form = set_password_form(user, request.POST)
			if form.is_valid():
				form.save()
				# authenticate the user
				p = form.cleaned_data.get('new_password1')
				u = authenticate(username=user.username, password=p)
				login(request, u)
				return HttpResponseRedirect(post_reset_redirect)
		else:
			form = set_password_form(None)
	else:
		context_instance['validlink'] = False
		form = None
	context_instance['form'] = form	   
	return render_to_response(template_name, context_instance=context_instance)
