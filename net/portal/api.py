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
from astrometry.net.portal.log import log
from astrometry.net import settings

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

    return HttpResponseJson({ 'status': 'success',
                              'message': 'authenticated user: ' + str(user.email),
                              })

def logout(request):
    return HttpResponse('Not implemented.')
