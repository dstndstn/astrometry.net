
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect
import shutil
import os, errno
import hashlib
import tempfile

def dashboard(request):
    return render_to_response("dashboard.html",
        {
		},
		context_instance = RequestContext(request))


@login_required
def get_api_key(request):
	try:
		prof = request.user.get_profile()
	except UserProfile.DoesNotExist:
		loginfo('Creating new UserProfile for', request.user)
		prof = UserProfile(user=request.user)
		prof.create_api_key()
		prof.save()
		
	return HttpResponse(str(prof))

class UploadFileForm(forms.Form):
	file  = forms.FileField()

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render_to_response('upload.html', {'form': form},
		context_instance = RequestContext(request))

def handle_uploaded_file(f):
    try:
        # get file onto disk
        file_hash = hashlib.sha1()
        temp_file_path = tempfile.mktemp()
        uploaded_file = open(temp_file_path, 'wb+')
        for chunk in f.chunks():
            uploaded_file.write(chunk)
            file_hash.update(chunk)
        uploaded_file.close()

        # move file into data directory
        file_directory = DiskFile.get_file_directory(file_hash.hexdigest())
        os.makedirs(file_directory)
        final_file_path = file_directory + file_hash.hexdigest()
        shutil.move(temp_file_path, final_file_path)
    except OSError as e:
        # we don't care if the directory already exists
        if e.errno == errno.EEXIST:
            pass
        else: raise

