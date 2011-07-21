from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django.core.urlresolvers import reverse
from django.template.loader import render_to_string

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.util import store_session_form
from log import *
from django import forms
from django.http import HttpResponseRedirect
import simplejson

class TagForm(forms.ModelForm):
    # so the primary key restriction isn't enforced
    text = forms.CharField(widget=forms.TextInput(attrs={'size':'30'}),
                           max_length=4096)
    class Meta:
        model = Tag
        exclude = ('text')

@login_required
def delete(req, category=None, recipient_id=None, tag_id=None):
    if category == 'user_image':
        tag = get_object_or_404(TaggedUserImage, user_image=recipient_id, tag=tag_id)
        if tag.tagger == req.user or tag.user_image.user == req.user:
            tag.delete()
    else:
        # TODO - do something useful?
        pass

    redirect_url = req.GET.get('next', '/')
    json = {'success': True}
    if req.is_ajax():
        response = simplejson.dumps(json)
        return HttpResponse(response, content_type='application/javascript')
    else:
        return HttpResponseRedirect(redirect_url)

@login_required
def new(req, category=None, recipient_id=None):
    if req.method == 'POST':
        recipient = None
        recipient_owner = None
        if category == 'user_image':
            recipient = get_object_or_404(UserImage, pk=recipient_id)
            recipient_owner = recipient.user
        form = TagForm(req.POST)
        redirect_url = req.POST.get('next','/')
        json = {}
        if form.is_valid():
            tag,created = Tag.objects.get_or_create(**form.cleaned_data)
            if category == 'user_image':
                tagged_user_image,created = TaggedUserImage.objects.get_or_create(
                    user_image=recipient,
                    tag=tag,
                    tagger=req.user
                )
            
            if req.is_ajax():
                form = TagForm()
                context = {
                    'tag': tag,
                    'category': category,
                    'recipient_id': recipient_id,
                    'recipient_owner': recipient_owner,
                    'next': redirect_url,
                }
                tag_html = render_to_string('tag/tag.html', context,
                                    context_instance=RequestContext(req))

                json['success'] = created
                json['tag_html'] = tag_html
        else:
            if req.is_ajax():
                json['success'] = False
            else:
                store_session_form(req.session, TagForm, req.POST)

        if req.is_ajax():
            context = {
                'tag_form': form,
                'category': category,
                'recipient_id': recipient_id,
                'recipient_owner': recipient_owner,
                'next': redirect_url,
            }
            form_html = render_to_string('tag/form.html', context,
                                context_instance=RequestContext(req))
            json['form_html'] = form_html
            
            response = simplejson.dumps(json)
            return HttpResponse(response, content_type='application/javascript')
        else:
            return redirect(redirect_url)
