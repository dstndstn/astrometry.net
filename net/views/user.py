import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.core.urlresolvers import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django.db.models import Count

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command
from astrometry.net.util import get_page, get_session_form, store_session_form
from astrometry.net.views.album import *
from astrometry.net.views.license import LicenseForm

class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        exclude = ('apikey', 'default_license')


def dashboard(request):
    return render_to_response("dashboard/base.html",
        {
        },
        context_instance = RequestContext(request))

@login_required
def save_profile(req):
    if req.method == 'POST':
        profile = req.user.get_profile()
        profile_form = ProfileForm(req.POST, instance=profile)
        license_form = LicenseForm(req.POST, instance=profile.default_license)
        
        if profile_form.is_valid() and license_form.is_valid():
            profile_form.save()
            license,created = License.objects.get_or_create(
                default_license=License.get_default(),
                allow_commercial_use=license_form.cleaned_data['allow_commercial_use'],
                allow_modifications=license_form.cleaned_data['allow_modifications'],
            )
            profile.default_license = license
            profile.save()

            messages.success(req, 'Profile successfully updated.')
        else:
            store_session_form(req.session, ProfileForm, req.POST)
            store_session_form(req.session, LicenseForm, req.POST)
            messages.error(req, 'Please fix the following errors:')
        return redirect('astrometry.net.views.user.dashboard_profile')

@login_required
def dashboard_profile(req):
    # user profile guaranteed to be created during openid login
    profile = req.user.get_profile()
           
    profile_form = get_session_form(req.session, ProfileForm, instance=profile)
    license_form = get_session_form(req.session, LicenseForm, instance=profile.default_license)
    context = {
        'profile_form': profile_form,
        'license_form': license_form,
        'profile': profile,
        'site_default_license': License.get_default(),
    }
    return render(req, "dashboard/profile.html", context)

@login_required
def dashboard_submissions(req):
    page_number = req.GET.get('page',1)
    page = get_page(req.user.submissions.all().order_by('-submitted_on'),15,page_number)
    context = {
        'submission_page': page
    }
    return render_to_response("dashboard/submissions.html",
        context,
        context_instance = RequestContext(req))

@login_required
def dashboard_user_images(req):
    page_number = req.GET.get('page',1)
    page = get_page(req.user.user_images.public_only(req.user),3*10,page_number)
    
    context = {
        'user':req.user,
        'image_page':page
    }
    
    return render_to_response('dashboard/user_images.html',
        context,
        context_instance = RequestContext(req))

@login_required
def dashboard_albums(req):
    page_number = req.GET.get('page',1)
    page = get_page(req.user.albums.all(),3*10,page_number)
    
    context = {
        'user': req.user,
        'album_page': page
    }
    
    return render_to_response('dashboard/albums.html',
        context,
        context_instance = RequestContext(req))

@login_required
def dashboard_create_album(req):
    album = Album(user=req.user)
    form = get_session_form(req.session, AlbumForm, instance=album)
        
    context = {
        'album_form': form,
    }
    return render(req, "dashboard/create_album.html", context)
    
def index(req, users=User.objects.all(),
          template_name='user/index.html', context={}):

    form = UserSearchForm(req.GET)
    if form.is_valid():
        query = form.cleaned_data.get('query')
        if query:
            users = users.filter(profile__display_name__icontains=query)

    sort = req.GET.get('sort', 'name')
    order = 'profile__display_name'
    if sort == 'name':
        #profiles = (UserProfilek.extra(select={'lower_name':'lower(profile.display_name)'})
        #             .order_by('lower_name'))
        order = 'profile__display_name'
    elif sort == 'date':
        order = 'date_joined'
    elif sort == 'images':
        users = users.annotate(Count('user_images'))
        order = '-user_images__count'
    
    users = users.order_by(order)
    page_number = req.GET.get('page',1)
    user_page = get_page(users,20,page_number)
    context.update({
        'user_page': user_page,
        'user_search_form': form,
    })
    return render(req, template_name, context)

class UserSearchForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'autocomplete':'off'}),
                                                  required=False)

def user_profile(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)

    context = {
        'display_user': user,
		'recent_images': user.user_images.public_only(req.user),	
        'recent_submissions': user.submissions.all().order_by('-submitted_on')[:10],
    }
    return render_to_response('user/profile.html',
        context,
        context_instance = RequestContext(req))

def user_images(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)

    page_number = req.GET.get('page',1)
    page = get_page(user.user_images.public_only(req.user),3*10,page_number)
    
    context = {
        'display_user': user,
        'image_page': page
    }
    
    return render_to_response('user/user_images.html',
        context,
        context_instance = RequestContext(req))

def user_albums(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)

    page_number = req.GET.get('page',1)
    page = get_page(user.albums.all(),3*10,page_number)
    
    context = {
        'display_user': user,
        'album_page': page
    }
    
    return render_to_response('user/albums.html',
        context,
        context_instance = RequestContext(req))

def user_submissions(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)

    page_number = req.GET.get('page',1)
    page = get_page(user.submissions.all().order_by('-submitted_on'),15,page_number)
    
    context = {
        'display_user': user,
        'submission_page': page
    }
    return render_to_response("user/submissions.html",
        context,
        context_instance = RequestContext(req))

def user_autocomplete(req):
    name = req.GET.get('q','')
    users = User.objects.filter(profile__display_name__icontains=name)[:8]
    response = HttpResponse(mimetype='text/plain')
    for user in users:
        response.write(user.get_profile().display_name + '\n')
    return response

