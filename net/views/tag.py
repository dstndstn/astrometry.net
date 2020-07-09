from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.template.loader import render_to_string
from django.db.models import Count

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.util import get_page, store_session_form
from astrometry.net.log import *
from django import forms
from django.http import HttpResponseRedirect
import json

class TagForm(forms.ModelForm):
    # so the primary key restriction isn't enforced
    text = forms.CharField(widget=forms.TextInput(attrs={'size':'30'}),
                           max_length=4096)
    class Meta:
        model = Tag
        exclude = ('text',)

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
    J = {'success': True}
    if req.is_ajax():
        response = json.dumps(J)
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
        J = {}
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
                tag_html = render_to_string('tag/tag.html', context, req)

                J['success'] = created
                J['tag_html'] = tag_html
        else:
            if req.is_ajax():
                J['success'] = False
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
            form_html = render_to_string('tag/form.html', context, req)
            J['form_html'] = form_html
            
            response = json.dumps(J)
            return HttpResponse(response, content_type='application/javascript')
        else:
            return redirect(redirect_url)

class TagSearchForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'autocomplete':'off'}),
                                                  required=False)
    exact = forms.BooleanField(initial=False, required=False)

def index(req, tags=Tag.objects.all(), 
          template_name='tag/index.html', context={}):

    logmsg('tags:', tags.count())

    form = TagSearchForm(req.GET)
    if form.is_valid():
        query = form.cleaned_data.get('query')
        if query:
            tags = tags.filter(text__icontains=query)
            logmsg('query for "%s":' % query, tags.count())
        else:
            logmsg('no query')

    tags = tags.annotate(Count('user_images'))
    sort = req.GET.get('sort', 'freq')
    order = '-user_images__count'
    if sort == 'name':
        order = 'text'
    elif sort == 'freq':
        tags = tags.annotate(Count('user_images'))
        order = '-user_images__count'
    
    tags = tags.order_by(order)
    logmsg('tags:', tags.count())
    page_number = req.GET.get('page', 1)
    page = get_page(tags, 50, page_number)
    context.update({
        'tag_page': page,
        'tags': tags,
        'tag_search_form': form,
    })
    return render(req, template_name, context)

def tag_autocomplete(req):
    name = req.GET.get('q','')
    tags = Tag.objects.filter(text__icontains=name)[:8]
    response = HttpResponse(''.join([t.text+'\n' for t in tags]), content_type='text/plain')
    return response
