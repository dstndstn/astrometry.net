from django.contrib.auth import authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

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


def login(request):
    json = request.POST.get('request-json')
    if not json:
        return HttpResponseBadRequest('no json')
    args = json2python(json)
    if args is None:
        return HttpResponseBadRequest('failed to parse JSON: ', json)
    uname = args.get('username')
    password = args.get('password')
    if uname is None or password is None:
        return HttpResponseBadRequest('need "username" and "password".')

    user = authenticate(username=uname, password=password)
    if user is None:
        return HttpResponseBadRequest('incorrect username/password')

    return HttpResponse('authenticated user: ' + str(user.email))

    #return HttpResponse('json: ' + str(json) + '\nargs: ' + str(args))


def logout(request):
    return HttpResponse('Not implemented.')
