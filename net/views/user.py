import os

from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.db.models import Count

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from django import forms
from django.http import HttpResponseRedirect

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command
from astrometry.net.util import get_page, get_session_form, store_session_form
from astrometry.net.views.album import *
from astrometry.net.views.license import LicenseForm
from astrometry.net.util import NoBulletsRadioSelect

from astrometry.net.views.human import human_required

class ProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        exclude = ('apikey', 'default_license', 'user')
        widgets = {
            'default_publicly_visible': NoBulletsRadioSelect(),
            }

def dashboard(request):
    return render(request, "dashboard/index.html")

@login_required
def save_profile(req):
    if req.method == 'POST':
        profile = get_user_profile(req.user)
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
    profile = get_user_profile(req.user)
           
    profile_form = get_session_form(req.session, ProfileForm, instance=profile)
    license_form = get_session_form(req.session, LicenseForm, instance=profile.default_license)
    context = {
        'profile_form': profile_form,
        'license_form': license_form,
        'profile': profile,
        'site_default_license': License.get_default(),
        'profile': profile,
    }
    return render(req, "dashboard/profile.html", context)

@login_required
def dashboard_submissions(req):
    page_number = req.GET.get('page',1)
    page = get_page(req.user.submissions.all().order_by('-submitted_on'),15,page_number)
    context = {
        'submission_page': page
    }
    return render(req, "dashboard/submissions.html", context)

@login_required
def dashboard_user_images(req):
    page_number = req.GET.get('page',1)
    page = get_page(req.user.user_images.public_only(req.user),3*10,page_number)
    context = {
        'user':req.user,
        'image_page':page
    }
    return render(req, 'dashboard/user_images.html', context)

@login_required
def dashboard_albums(req):
    page_number = req.GET.get('page',1)
    page = get_page(req.user.albums.all(),3*10,page_number)
    context = {
        'user': req.user,
        'album_page': page
    }
    return render(req, 'dashboard/albums.html', context)

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
        'profile': get_user_profile(req.user),
    }
    return render(req, 'user/profile.html', context)

def user_images(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)
    page_number = req.GET.get('page',1)
    page = get_page(user.user_images.public_only(req.user),3*10,page_number)
    context = {
        'display_user': user,
        'image_page': page,
    }
    return render(req, 'user/user_images.html', context)

def user_albums(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)
    page_number = req.GET.get('page',1)
    page = get_page(user.albums.all(),3*10,page_number)
    context = {
        'display_user': user,
        'album_page': page
    }
    return render(req, 'user/albums.html', context)

def user_submissions(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)
    page_number = req.GET.get('page',1)
    page = get_page(user.submissions.all().order_by('-submitted_on'),15,page_number)
    context = {
        'display_user': user,
        'submission_page': page
    }
    return render(req, "user/submissions.html", context)

def user_autocomplete(req):
    name = req.GET.get('q','')
    users = User.objects.filter(profile__display_name__icontains=name)[:8]
    response = HttpResponse(mimetype='text/plain')
    for user in users:
        pro = get_user_profile(user)
        response.write(pro.display_name + '\n')
    return response

