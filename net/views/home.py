from __future__ import print_function
from django.shortcuts import redirect, render
from django.db.models import Count
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
import os

if __name__ == '__main__':
    os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
    import django
    django.setup()

from astrometry.net.models import *

def home(req):
    from astrometry.net import settings

    # home-brewed {% cycle %} replacement
    class nexthost(object):
        def __init__(self, hosts):
            self.hosts = hosts
            self.i = 0
        def get(self):
            rtn = self.hosts[self.i]
            self.i = (self.i + 1) % len(self.hosts)
            return rtn

    context = {
        'uimages':UserImage.objects.public_only(req.user)[:8],
    }
    if hasattr(settings,"MULTI_HOSTS") and len(settings.MULTI_HOSTS):
        context.update({
            'multi_hosts': nexthost(settings.MULTI_HOSTS),
        })
    return render(req, 'home.html', context)

def maintenance(req):
    return render(req, 'maintenance.html', {})

def support(req):
    context = {}
    return render(req, 'support.html', context)

def api_help(req):
    context = dict(user_profile=get_user_profile(req.user))
    return render(req, 'api_help.html', context)

# @login_required
# def new_api_key(req):
#     pro = get_user_profile(req.user)
#
#     return HttpResponse('you are ' + str(req.user) + 'with profile' + str(pro)
#                         + 'or', str(req.user.get_profile()))

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


def signin(req):
    from astrometry.net import settings
    from social_core.backends.utils import load_backends

    ctxt = {}
    ctxt.update({
            'available_backends': load_backends(settings.AUTHENTICATION_BACKENDS)
            })
    return render(req, 'signin.html', ctxt)

def signout(request):
    """Logs out user"""
    from django.contrib.auth import logout
    logout(request)
    return redirect('/')


from django.contrib.auth.decorators import login_required

@login_required
def signedin(req):
    print('signedin.')
    user = req.user
    if not user.is_authenticated:
        return HttpResponse('not authenticated: ' + user)

    if user is not None:
        if user.is_active:
            try:
                profile = user.profile
            except UserProfile.DoesNotExist:
                loginfo('Creating new UserProfile for', user)
                profile = UserProfile(user=user)
                profile.create_api_key()
                profile.create_default_license()
                if user.get_full_name():
                    profile.display_name = user.get_full_name()
                else:
                    profile.display_name = user.username
                profile.save()
            return redirect('/dashboard/')
        else:
            return HttpResponse('Disabled account')
    else:
        return HttpResponse('Unknown user')

@login_required
def newuser(req):
    #print 'New user'
    #return HttpResponse('hello, new guy')
    return HttpResponseRedirect('/')




def about_user(backend, user, response, *args, **kwargs):
    print('Backend:', backend)
    print('backend name:', backend.name)
    print('user:', user)
    #print 'User profile:', user.get_profile()
    try:
        print('user profile:', user.profile)
    except:
        pass

    print('Existing users:')
    from django.contrib.auth.models import User
    U = User.objects.all()
    for u in U:
        print('  ', u.username, u.email)

    print('Login response:') #, response
    keys = response.keys()
    keys.sort()
    for k in keys:
        print('  ', k, '=', response[k])

    print('args:', args)
    print('kwargs:', kwargs.keys())

    for k in ['username', 'is_new', 'uid', 'new_association', 'social', 'details']:
        if k in kwargs:
            print(k, ':', kwargs[k])


    # if backend.name == 'facebook':
    #     profile = user.get_profile()
    #     if profile is None:
    #         profile = Profile(user_id=user.id)
    #     profile.gender = response.get('gender')
    #     profile.link = response.get('link')
    #     profile.timezone = response.get('timezone')
    #     profile.save()


def load_user(*args, **kwargs):
    print()
    print('load_user()')
    about_user(*args, **kwargs)
    print()
    rtn = {}
    try:
        #print 'kwargs:', kwargs.keys()
        #backend, user, response = args[:3]
        backend = kwargs['backend']
        response = kwargs['response']

        if backend.name == 'google':
            email = response['emails'][0]['value']
        elif backend.name == 'github':
            email = response['email']
        print('Email:', email)

        from django.contrib.auth.models import User
        U = User.objects.all()
        try:
            #u = U.get(email=email)
            u = U.filter(email=email)[0]
        except:
            import traceback
            print('caught:')
            traceback.print_exc()
        print('Found existing user:', u)
        rtn['user'] = u
    except:
        import traceback
        print('caught:')
        traceback.print_exc()
        pass
    return rtn

def pre_get_username(*args, **kwargs):
    print()
    print('pre_get_username()')
    about_user(*args, **kwargs)
    print()

def post_get_username(*args, **kwargs):
    print()
    print('post_get_username()')
    about_user(*args, **kwargs)
    print()

def post_create_user(*args, **kwargs):
    print()
    print('post_create_user()')
    about_user(*args, **kwargs)
    print()

def post_auth(*args, **kwargs):
    print()
    print('post_auth()')
    about_user(*args, **kwargs)
    print()

if __name__ == '__main__':
    from django.test import Client
    c = Client()
    #r = c.get('/login/twitter/')
    r = c.get('/api_help')
    with open('out.html', 'wb') as f:
        for x in r:
            f.write(x)
