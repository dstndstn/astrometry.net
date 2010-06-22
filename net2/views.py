from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render_to_response, get_object_or_404
from django.core.context_processors import csrf
from django.template import Context, loader, RequestContext

from django import forms

class LoginForm(forms.Form):
	openid = forms.CharField()


def login(req):
	if req.method == 'POST':
		form = LoginForm(req.POST)
		if form.is_valid():
			openid = form.cleaned_data.get('openid', None)
			#return HttpResponseRedirect('/thanks/') # Redirect after POST
	else:
		form = LoginForm()

	c = {'form': form,
		 'next': '/home'}
	c.update(csrf(req))
	return render_to_response('login.html', c)

def logout(req):
	return HttpResponse('not implemented')
