from django.contrib import auth
from django.http import HttpResponse, HttpResponseBadRequest
from django.core.exceptions import ObjectDoesNotExist
from django.views.decorators.csrf import csrf_exempt

from api_util import *
from userprofile import *
from log import *

json_type = 'text/plain' # 'application/json'


class HttpResponseErrorJson(HttpResponse):
        def __init__(self, errstring):
                args = { 'status': 'error',
                                 'errormessage': errstring }
                doc = python2json(args)
                super(HttpResponseErrorJson, self).__init__(doc, content_type=json_type)

class HttpResponseJson(HttpResponse):
        def __init__(self, args):
                doc = python2json(args)
                super(HttpResponseJson, self).__init__(doc, content_type=json_type)

def create_session(key):
        from django.conf import settings
        engine = __import__(settings.SESSION_ENGINE, {}, {}, [''])
        sess = engine.SessionStore(key)
        # sess.session_key creates a new key if necessary.
        return sess

# decorator for extracting JSON arguments from a POST.
def requires_json_args(handler):
        def handle_request(request, *pargs, **kwargs):
                loginfo('POST: ' + str(request.POST))
                json = request.POST.get('request-json')
                loginfo('JSON: "%s"' % json)
                if not json:
                        return HttpResponseErrorJson('no json')
                args = json2python(json)
                if args is None:
                        return HttpResponseErrorJson('failed to parse JSON: "%s"' % json)
                request.json = args
                #print 'json:', request.json
                return handler(request, *pargs, **kwargs)
        return handle_request

# decorator for retrieving the user's session based on a session key in JSON.
# requires "request.json" to exist: you probably need to precede this decorator
# by the "requires_json_args" decorator.
def requires_json_session(handler):
        def handle_request(request, *args, **kwargs):
                #print 'requires_json_session decorator running.'
                if not 'session' in request.json:
                        return HttpResponseErrorJson('no "session" in JSON.')
                key = request.json['session']
                session = create_session(key)
                if not session.exists(key):
                        return HttpResponseErrorJson('no session with key "%s"' % key)
                request.session = session
                #print 'session:', request.session
                resp = handler(request, *args, **kwargs)
                session.save()
                # remove the session from the request so that SessionMiddleware
                # doesn't try to set cookies.
                del request.session
                return resp
        return handle_request

def requires_json_login(handler):
        def handle_request(request, *args, **kwargs):
                #print 'requires_json_login decorator running.'
                user = auth.get_user(request)
                #print 'user:', request.session
                if not user.is_authenticated():
                        return HttpResponseErrorJson('user is not authenticated.')
                return handler(request, *args, **kwargs)
        return handle_request


@csrf_exempt
@requires_json_args
@requires_json_session
def api_upload(request):
    return HttpResponse('hello')

'''
    logmsg('request:' + str(request))
    logmsg('api_upload: got request: ' + str(request.FILES['file'].size))
    return HttpResponse('hello')
'''

@csrf_exempt
@requires_json_args
def api_login(request):
        apikey = request.json.get('apikey')
        if apikey is None:
                return HttpResponseErrorJson('need "apikey"')

        loginfo('Got API key:', apikey)
        try:
                profile = UserProfile.objects.all().get(apikey=apikey)
                loginfo('Matched API key:', profile)
        except ObjectDoesNotExist:
                loginfo('Bad API key')
                return HttpResponseErrorJson('bad apikey')

        # Successful API login.  Register on the django side.

        # can't do this:
        #password = profile.user.password
        #auth.authenticate(username=profile.user.username, password=password)
        #auth.login(request, profile.user)

        loginfo('backends:' + str(auth.get_backends()))

        user = profile.user
        user.backend = 'django_openid_auth.auth.OpenIDBackend'

        session = create_session(None)

        request.session = session
        auth.login(request, profile.user)
        # so that django middleware doesnt' try to set cookies in response
        del request.session
        session.save()

        key = session.session_key
        loginfo('Returning session key ' + key)
        return HttpResponseJson({ 'status': 'success',
                                                          'message': 'authenticated user: ' + str(profile.user.email),
                                                          'session': key,
                                                          })

