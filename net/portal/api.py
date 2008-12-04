from django.contrib import auth
from django.contrib.auth import authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

# from django.contrib.sessions.backends.db import SessionStore
# from django.contrib.sessions.models import Session

from django.http import HttpResponse, HttpResponseBadRequest #HttpResponseRedirect, QueryDict
#from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse
#from django.core.exceptions import ObjectDoesNotExist

#from astrometry.net.portal.job import Job, Submission, DiskFile, Tag
from astrometry.net.portal.log import log as logmsg
from astrometry.net import settings

import os
import pickle
import base64
from Crypto.PublicKey import RSA

def json2python(json):
    from simplejson import loads
    try:
        return loads(json)
    except:
        pass
    return None

def python2json(py):
    from simplejson import dumps
    return dumps(py)

json_type = 'text/plain' # 'application/json'

class HttpResponseErrorJson(HttpResponseBadRequest):
    def __init__(self, errstring):
        args = { 'status': 'error',
                 'errormessage': errstring }
        doc = python2json(args)
        super(HttpResponseErrorJson, self).__init__(doc, content_type=json_type)

class HttpResponseJson(HttpResponse):
    def __init__(self, args):
        doc = python2json(args)
        super(HttpResponseJson, self).__init__(doc, content_type=json_type)

#def json_auth_token_decorator(func):

def create_session(key):
    from django.conf import settings
    engine = __import__(settings.SESSION_ENGINE, {}, {}, [''])
    sess = engine.SessionStore(key)
    # sess.session_key creates a new key if necessary.
    return sess

def login(request):
    json = request.POST.get('request-json')
    if not json:
        return HttpResponseErrorJson('no json')
    args = json2python(json)
    if args is None:
        return HttpResponseErrorJson('failed to parse JSON: ', json)
    uname = args.get('username')
    password = args.get('password')
    if uname is None or password is None:
        return HttpResponseErrorJson('need "username" and "password".')

    user = authenticate(username=uname, password=password)
    if user is None:
        return HttpResponseErrorJson('incorrect username/password')

    # User has successfully logged in.  Create session.
    session = create_session(None)
    #request.json_session = session

    request.session = session
    auth.login(request, user)
    del request.session

    # generate pubkey
    keybits = 512
    privkey = RSA.generate(keybits, os.urandom)
    privkeystr = base64.encodestring(pickle.dumps(privkey.__getstate__()))
    pubkey = privkey.publickey()
    pubkeystr = base64.encodestring(pickle.dumps(pubkey.__getstate__()))

    # client encodes like this:
    message = 'Mary had a little hash'
    ckey = RSA.RSAobj()
    ckey.__setstate__(pickle.loads(base64.decodestring(pubkeystr)))
    secret = ckey.encrypt(message, K='')
    logmsg('secret = ', secret)

    # server decodes like this:
    skey = RSA.RSAobj()
    skey.__setstate__(pickle.loads(base64.decodestring(privkeystr)))
    dmessage = skey.decrypt(secret)
    logmsg('decrypted = ', dmessage)

    session['private_key'] = privkeystr
    session['public_key'] = pubkeystr

    session.save()

    key = session.session_key
    return HttpResponseJson({ 'status': 'success',
                              'message': 'authenticated user: ' + str(user.email),
                              'session': key,
                              'pubkey': pubkeystr,
                              })

def amiloggedin(request):
    json = request.POST.get('request-json')
    if not json:
        return HttpResponseErrorJson('no json')
    args = json2python(json)
    if args is None:
        return HttpResponseErrorJson('failed to parse JSON: ', json)
    if not 'session' in args:
        return HttpResponseJson({ 'loggedin': False,
                                  'reason': 'no session in args'})
    key = args['session']
    session = create_session(key)
    if not session.exists(key):
        return HttpResponseJson({ 'loggedin': False,
                                  'reason': 'no such session with key: '+key})
    request.session = session
    user = auth.get_user(request)
    del request.session

    if not user.is_authenticated():
        return HttpResponseJson({ 'loggedin': False,
                                  'reason': 'user is not authenticated.'})
    return HttpResponseJson({ 'loggedin': True,
                              'username': user.username})

def logout(request):
    return HttpResponse('Not implemented.')
