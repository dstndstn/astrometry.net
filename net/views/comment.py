from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect

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
    recipient = get_object_or_404(Commentable, pk=recipient_id)
    if req.method == 'POST':
        author = req.user
        comment = Comment(
            author=author,
            recipient=recipient,
            text=req.POST['text']
        )
        comment.save()
        redirect_url = req.POST['next']
        if redirect_url == None:
            redirect_url = '/'
        return HttpResponseRedirect(redirect_url)
    else:
        # show a generic comment form
        pass
