import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms
from django.http import HttpResponseRedirect
from django.contrib import messages

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net.util import NoBulletsRenderer, get_page, get_session_form, store_session_form

from astrometry.util.run_command import run_command

from astrometry.net.views.comment import *


def album(req, album_id=None):
    album = get_object_or_404(Album, pk=album_id)

    comment_form = get_session_form(req.session, PartialCommentForm)

    page_number = req.GET.get('page',1)
    page = get_page(album.user_images.public_only(req.user),4*3,page_number)
    context = {
        'album': album,
        'comment_form': comment_form,
        'image_page': page,
        'request': req,
    }

    if album.is_public() or (album.user == req.user and req.user.is_authenticated()):
        template = 'album/view.html'
    #elif SharedHideable.objects.filter(shared_with=req.user.id, hideable=album).count():
    #    template = 'album/view.html'
    else:
        messages.error(req, "Sorry, you don't have permission to view this content.")
        template = 'album/permission_denied.html'
    return render(req, template, context)

class AlbumForm(forms.ModelForm):
    class Meta:
        model = Album
        exclude = ('user', 'owner', 'user_images', 'tags', 'created_at', 'comment_receiver')
        widgets = {
            'description': forms.Textarea(attrs={'cols':60,'rows':3}),
            'publicly_visible': forms.RadioSelect(renderer=NoBulletsRenderer)
        }

    def clean(self):
        cleaned_data = self.cleaned_data
        title = cleaned_data.get('title')
        if title:
            # make sure the user doesn't have another album with the same title
            query = Album.objects.filter(user=self.instance.user, title=title)
            query = query.exclude(pk=self.instance.id)
            if query.count() != 0:
                self._errors['title'] = self.error_class(['You already have an album with this title.'])
                del cleaned_data['title']

        return cleaned_data
        
@login_required
def edit(req, album_id=None):
    album = get_object_or_404(Album, pk=album_id) 
    if album.user != req.user:
        messages.error(req, "Sorry, you don't have permission to view this content.")
        return render(req, 'album/permission_denied.html')

    if req.method == 'POST':
        form = AlbumForm(req.POST, instance=album)
        if form.is_valid():
            form.save()
            messages.success(req, 'Album details successfully updated.')
            return redirect(album)
        else:
            messages.error(req, 'Please fix the following errors:')
    else:
        form = AlbumForm(instance=album)
        
    context = {
        'album_form': form,
        'album': album,
    }
    return render(req, 'album/edit.html', context)


@login_required
def new(req):
    if req.method == 'POST':
        album = Album(user=req.user)
        form = AlbumForm(req.POST, instance=album)
        if form.is_valid():
            form.save(commit=False)
            album.comment_receiver=CommentReceiver.objects.create()
            album.save()
            messages.success(req, "Album '%s' successfully created." % album.title)
            return redirect(album)
        else:
            store_session_form(req.session, AlbumForm, req.POST)
            messages.error(req, 'Please fix the following errors:')
            return redirect(req.POST.get('from','/'))
    else:
        pass

@login_required
def delete(req, album_id):
    album = get_object_or_404(Album, pk=album_id)
    redirect_url = req.GET.get('next','/')
    if album.user == req.user:
        album.delete()
        messages.success(req, "Album '%s' successfully deleted." % album.title)
        return HttpResponseRedirect(redirect_url)
    else:
        # render_to_response a "you don't have permission" view
        pass
