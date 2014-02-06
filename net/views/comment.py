from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django.template.loader import render_to_string

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.util import store_session_form
from astrometry.net.log import *
from django import forms
from django.http import HttpResponseRedirect
import simplejson

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment

class PartialCommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        exclude = ('recipient', 'author')
        widgets = {'text':forms.Textarea(attrs={'cols':60,'rows':3})}

@login_required
def new(req, category=None, recipient_id=None):
    recipient = get_object_or_404(CommentReceiver, pk=recipient_id)
    if req.method == 'POST':
        form = PartialCommentForm(req.POST)
        redirect_url = req.POST.get('next', '/')
        json = {}
        if form.is_valid():
            comment = form.save(commit=False)
            comment.recipient = recipient
            comment.author = req.user
            comment.save()
        
            if req.is_ajax():
                form = PartialCommentForm()
                context = {
                    'comment': comment,
                    'next': redirect_url,
                }
                comment_html = render_to_string('comment/comment.html', context,
                                    context_instance=RequestContext(req))
                json['success'] = True
                json['comment_html'] = comment_html
        else:
            if req.is_ajax():
                json['success'] = False
            else:
                store_session_form(req.session, PartialCommentForm, req.POST)

        if req.is_ajax():
            context = {
                'comment_form': form,
                'recipient_id': recipient.id,
                'category': category,
                'next': redirect_url,
            }
            form_html = render_to_string('comment/form.html', context,
                                context_instance=RequestContext(req))
            json['form_html'] = form_html
        
            response = simplejson.dumps(json)
            return HttpResponse(response, content_type='application/javascript')
        else:
            return redirect(redirect_url)
             
    else:
        # show a generic comment form
        pass

@login_required
def delete(req, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    redirect_url = req.GET.get('next','/')
    if comment.recipient.owner == req.user or comment.author == req.user:
        comment.delete()
        
        json = {'success': True}
        if req.is_ajax():
            response = simplejson.dumps(json)
            return HttpResponse(response, content_type='application/javascript')
        else:
            return HttpResponseRedirect(redirect_url)
    else:
        # render_to_response a "you don't have permission" view
        pass
