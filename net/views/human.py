import os
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms

if __name__ == '__main__':
    os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
    import django
    django.setup()
    
from django.contrib.auth.views import redirect_to_login

# Decorator
def human_required(real_view):
    def make_decorated(real_view):
        #print('human_required decorator')
        #print('real_view is', real_view)
        def view(req, **kwargs):
            #print('human_required decorated function')
            #print('req is', req)
            #print('real_view is', real_view)
            #print('session keys:', req.session.keys())
            if not 'human' in req.session:
                #req.session['human_redir'] = real_view
                #return ask_human(req)
                #print('No "human" session key')
                #print('full path of request:', req.get_full_path())
                #print('redirecting to login page...')
                return redirect_to_login(req.get_full_path(), login_url='/ask_human')
            #print('verified human')
            return real_view(req, **kwargs)
        return view
    return make_decorated(real_view)

def human_or_ref_required(real_view):
    def make_decorated(real_view):
        def view(req, **kwargs):
            print('Human or Referer: ref', req.headers.get('Referer'))
            if 'Referer' in req.headers:
                return real_view(req, **kwargs)
            if 'human' in req.session:
                return real_view(req, **kwargs)
            return redirect_to_login(req.get_full_path(), login_url='/ask_human')
        return view
    return make_decorated(real_view)

def ask_human(req):
    #print('are you human?')
    #print('req:', req)
    return render(req, 'ask_human.html', {'next': req.GET.get('next')})

def am_human(req):
    human = req.POST.get('human')
    human = (human == 'yup')
    if human:
        req.session['human'] = True
    next = req.POST.get('next')
    return HttpResponseRedirect(next)

@human_required
def test_human(req):
    return HttpResponse('Test page -- human required')

@human_or_ref_required
def test_human_2(req, user_image_id=None):
    return HttpResponse('Test page -- human required -- uid ' + user_image_id)

def not_human(req):
    if 'human' in req.session:
        del req.session['human']
    return HttpResponse('bleep blorp')

def poison(req, depth=None, num=None):
    from random import randint
    if depth is None:
        depth = 1
    depth = int(depth)
    ltxt = '\n'.join(['<li><a href="/poison/%i/%i">%i</a></li>' % (depth+1, randint(0, 1000),
                                                                   randint(0, 1000))
                      for i in range(3)])
    txt = '''
    <html><head><title>Poisoned well</title></head>
    <body>
    <h1>Drink from the well!</h1>
    <p>Under construction, but you can dig deeper here:
    <ul>
    %s
    </ul>
    </p>
    </body>
    </html>
    ''' % ltxt
    return HttpResponse(txt)

if __name__ == '__main__':
    from django.test import Client
    c = Client()
    #r = c.get('/test', follow=True)

    r = c.get('/poison/1/1')
    print('Got', r)
    with open('out.html', 'wb') as f:
        for x in r:
            f.write(x)
    import sys
    sys.exit(0)

    if True:
        r = c.get('/test2/11151437', follow=True)
        print('Response:', r)
        with open('out.html', 'wb') as f:
            for x in r:
                f.write(x)
        r = c.post('/am_human', { "human":"yup", "next":"/test2/11151437" }, follow=True)
        print('Response 2:', r)
        print('Cookies:', c.cookies)
        print('Session:', c.session)
        print('Session:', c.session.items())
        with open('out2.html', 'wb') as f:
            for x in r:
                f.write(x)

    c = Client()
    r = c.get('/test2/11151437', follow=True, headers={'Referer':'/'})
    print('Response 3:', r)
    with open('out3.html', 'wb') as f:
        for x in r:
            f.write(x)
    
    #r = c.get('/extraction_image_full/5906850')


    
