from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext
from django import forms
from astrometry.net.models import *
from astrometry.net.util import get_page, get_session_form, NoBulletsRenderer

class ImageSearchForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'autocomplete':'off'}),
                                                  required=False)

def images(req):
    images = UserImage.objects.all()
    form = ImageSearchForm(req.GET)
    if form.is_valid():
        query = form.cleaned_data.get('query')
        if query:
            images = images.filter(tags__text__icontains=query)

    images = images.order_by('submission__submitted_on')
    page_number = req.GET.get('page',1)
    image_page = get_page(images,3*10,page_number)
    context = {
        'image_search_form': form,
        'image_page': image_page,
    }
    return render(req, 'search/user_images.html', context)

class UserSearchForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'autocomplete':'off'}),
                                                  required=False)

def users(req):
    users = User.objects.all()
    form = UserSearchForm(req.GET)
    if form.is_valid():
        query = form.cleaned_data.get('query')
        if query:
            users = users.filter(profile__display_name__icontains=query)

    users = users.order_by('profile__display_name')
    page_number = req.GET.get('page',1)
    user_page = get_page(users,20,page_number)
    context = {
        'user_search_form': form,
        'user_page': user_page,
    }
    return render(req, 'search/users.html', context)

