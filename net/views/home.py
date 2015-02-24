from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext
from django.db.models import Count
from astrometry.net.models import *

def home(req):
    context = {
        'images':UserImage.objects.public_only(req.user),
    }
    return render(req, 'home.html', context)

def support(req):
    context = {}
    return render(req, 'support.html', context)

def api_help(req):
    context = {}
    return render(req, 'api_help.html', context)
    
def explore(req):
    recent_images = UserImage.objects.public_only(req.user)[:12]
    recent_comments = Comment.objects.filter(recipient__userimage__isnull=False)[:8]
    newest_users = User.objects.all().order_by('-date_joined')[:10]
    popular_tags = (Tag.objects.all().annotate(Count('user_images'))
                                     .order_by('-user_images__count')[:35])
    context = {
        'recent_images': recent_images,
        'newest_users': newest_users,
        'popular_tags': popular_tags,
		'recent_comments': recent_comments,
    }
    return render(req, 'explore.html', context)
    

def login(req):
    #from social_auth import __version__ as version
    from social import __version__ as version

    from astrometry.net import settings
    from social.backends.utils import load_backends

    ctxt = RequestContext(req)

    print 'Backends:', ctxt['backends']

    ctxt.update({
            #'plus_id': getattr(settings, 'SOCIAL_AUTH_GOOGLE_PLUS_KEY', None),
            #'plus_scope': ' '.join(GooglePlusAuth.DEFAULT_SCOPE),
            'available_backends': load_backends(settings.AUTHENTICATION_BACKENDS)
            })
    # print 'Context:', ctxt
    # stack = []
    # while True:
    #     try:
    #         d = ctxt.pop()
    #     except:
    #         break
    #     print 'Keys:', d.keys()
    #     stack.append(d)
    # while len(stack):
    #     d = stack.pop()
    #     ctxt.push(d)

    return render_to_response('login.html', {'version': version},
                              ctxt)


def logout(request):
    from django.contrib.auth import logout as auth_logout
    """Logs out user"""
    auth_logout(request)
    return redirect('/')
