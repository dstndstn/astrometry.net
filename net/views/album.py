import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms
from django.http import HttpResponseRedirect
from django.contrib import messages

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from astrometry.net.tmpfile import *

from astrometry.util.run_command import run_command

from astrometry.net.views.comment import *
from astrometry.net.util import get_page


def album(req, album_id=None):
    album = get_object_or_404(Album, pk=album_id)
    comment_form = PartialCommentForm()

    page_number = req.GET.get('page',1)
    page = get_page(album.user_images.all(),4*3,page_number)
    context = {
        'album': album,
        'comment_form': comment_form,
        'image_page': page,
        'request': req,
    }

    if album.is_public() or (album.user == req.user and req.user.is_authenticated()):
        template = 'album/view.html'
    elif SharedHideable.objects.filter(shared_with=req.user.id, hideable=image).count():
        template = 'album/view.html'
    else:
        messages.error(req, "Sorry, you don't have permission to view this content")
        template = 'permission_denied.html'
    return render_to_response(template, context,
        context_instance = RequestContext(req))

