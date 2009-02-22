from django.http import *
from django.template import Context, RequestContext, loader
from django import forms as forms
from django.core.urlresolvers import reverse
from django.forms import ValidationError

from models import UploadedFile
import logging
import os.path

from astrometry.net import settings

logfile = settings.PORTAL_LOGFILE
logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s %(levelname)s %(message)s',
					filename=logfile,
					)

class UploadIdField(forms.CharField):
	def clean(self, value):
		val = super(UploadIdField, self).clean(value)
		if not val:
			return val
		try:
			up = UploadedFile.objects.get(uploadid=val)
		except:
			raise ValidationError('That upload ID wasn\'t uploaded')
		path = up.get_filename()
		if not os.path.exists(path):
			raise ValidationError('No file for that upload id')
		return up

class UploadForm(forms.Form):
	upload_id = forms.CharField(max_length=32, widget=forms.HiddenInput)
	file = forms.FileField(widget=forms.FileInput(attrs={'size':'40'}))

def get_ajax_html(name=''):
	ctxt = {
		'name' : name,
		}
	t = loader.get_template('upload/progress-ajax-meter.html')
	return t.render(Context(ctxt))

def get_ajax_javascript(name=''):
	ctxt = {
		'name' : name,
		'xmlurl' : reverse(progress_xml) + '?upload_id='
		}
	t = loader.get_template('upload/progress-ajax-meter.js')
	return t.render(Context(ctxt))

def progress_ajax(request):
	if not request.GET:
		return HttpResponse('no GET')
	if not 'upload_id' in request.GET:
		return HttpResponse('no upload_id')
	id = request.GET['upload_id']
	logging.debug("Upload progress meter for id %s" % id)

	html = get_ajax_html()
	js = get_ajax_javascript()

	ctxt = {
		'javascript' : js,
		'meter' : html,
		'id' : id,
		}
	t = loader.get_template('upload/progress-ajax.html')
	c = RequestContext(request, ctxt)
	return HttpResponse(t.render(c))

def uploadform(request, template_name='upload/upload.html', onload=None,
			   target=None):
	#logging.debug("Upload form request.");
	id = UploadedFile.generateId()
	form = UploadForm({'upload_id': id})
	if target is None:
		target = settings.UPLOADER_URL
	ctxt = {
		'form' : form,
		'action' : target,
		'onload': onload,
		}
	t = loader.get_template(template_name)
	c = Context(ctxt)
	return HttpResponse(t.render(c))

def progress_xml(request):
	logging.debug('XML request')
	if not request.GET:
		return HttpResponseBadRequest('no GET')
	if not 'upload_id' in request.GET:
		return HttpResponseBadRequest('no upload_id')
	id = request.GET['upload_id']
	logging.debug('XML request for id ' + id)
	ups = UploadedFile.objects.all().filter(uploadid=id)
	if not len(ups):
		return HttpResponseNotFound('no such id')
	up = ups[0]
	tag = up.xml()
	if not len(tag):
		return HttpResponse('no tag')
	res = HttpResponse()
	res['Content-type'] = 'text/xml'
	res.write(tag)
	return res

