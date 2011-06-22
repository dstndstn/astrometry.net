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
        #return HttpResponseRedirect(redirect_url)
        # HACK
        return render_to_response('redirect.html', 
                                  {'url': redirect_url},
                                  context_instance = RequestContext(req))
    else:
        # show a generic comment form
        pass

@login_required
def delete(req, comment_id):
    comment = get_object_or_404(Comment, pk=comment_id)
    redirect_url = req.GET['next']
    if redirect_url == None:
        redirect_url = '/'
    if comment.recipient.owner == req.user or comment.author == req.user:
        comment.delete()
        #return HttpResponseRedirect(redirect_url)
        # HACK
        return render_to_response('redirect.html', 
                                  {'url': redirect_url},
                                  context_instance = RequestContext(req))

    else:
        # render_to_response a "you don't have permission" view
        pass
