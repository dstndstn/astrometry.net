import base64

from functools import wraps

from django.contrib import auth
from django.shortcuts import get_object_or_404
from django.http import HttpResponse, HttpResponseBadRequest
from django.core.exceptions import ObjectDoesNotExist
from django.views.decorators.csrf import csrf_exempt

# astrometry.net imports
from astrometry.net.models import *
from astrometry.net.views.submission import handle_upload
from api_util import *
from userprofile import *
from log import *
from tmpfile import *
import settings

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
    @wraps(handler)
    def handle_request(request, *pargs, **kwargs):
        #loginfo('POST: ' + str(request.POST))
        json = request.POST.get('request-json')
        loginfo('POST: JSON: "%s"' % json)
        if not json:
            loginfo('POST: keys: ', request.POST.keys())
            loginfo('GET keys: ', request.GET.keys())
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
    @wraps(handler)
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
    @wraps(handler)
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
    #logmsg('request:' + str(request))
    upfile = request.FILES.get('file', None)
    logmsg('api_upload: got file', upfile)
    logmsg('request.POST has keys:', request.POST.keys())
    logmsg('request.GET has keys:', request.GET.keys())
    logmsg('request.FILES has keys:', request.FILES.keys())
    #logmsg('api_upload: got request: ' + str(request.FILES['file'].size))
    logmsg('received files:')
    
    df = handle_upload(file=request.FILES['file'])
    submittor = request.user if request.user.is_authenticated() else None
    sub = Submission(user=submittor, disk_file=df, scale_type='ul', scale_units='degwidth')
    sub.save()

    return HttpResponseJson({'status': 'success',
                             'subid': sub.id,
                             'hash': sub.disk_file.file_hash}) 

@csrf_exempt
@requires_json_args
@requires_json_session
def url_upload(req):
    logmsg('request:' + str(req))
    url = req.json.get('url')
    logmsg('url: %s' % url)

    df = handle_upload(url=url)
    submittor = req.user if req.user.is_authenticated() else None
    sub = Submission(user=submittor, disk_file=df, url=url, scale_type='ul', scale_units='degwidth')
    sub.save()

    return HttpResponseJson({'status': 'success',
                             'subid': sub.id,
                             'hash': sub.disk_file.file_hash}) 

def write_wcs_file(req, wcsfn):
    from astrometry.util import util as anutil
    wcsparams = []
    wcs = req.json['wcs']
    for name in ['crval1', 'crval2', 'crpix1', 'crpix2',
                 'cd11', 'cd12', 'cd21', 'cd22', 'imagew', 'imageh']:
        wcsparams.append(wcs[name])
    wcs = anutil.Tan(*wcsparams)
    wcs.write_to(wcsfn)


@csrf_exempt
@requires_json_args
@requires_json_session
def api_sdss_image_for_wcs(req):
    from sdss_image import plot_sdss_image
    wcsfn = get_temp_file()
    plotfn = get_temp_file()
    write_wcs_file(req, wcsfn)
    plot_sdss_image(wcsfn, plotfn)
    return HttpResponseJson({'status': 'success',
                             'plot': base64.b64encode(open(plotfn).read()),
                             })

@csrf_exempt
@requires_json_args
@requires_json_session
def api_galex_image_for_wcs(req):
    from galex_jpegs import plot_into_wcs
    wcsfn = get_temp_file()
    plotfn = get_temp_file()
    write_wcs_file(req, wcsfn)
    plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR)
    return HttpResponseJson({'status': 'success',
                             'plot': base64.b64encode(open(plotfn).read()),
                             })

@csrf_exempt
@requires_json_args
@requires_json_session
def api_submission_images(req):
    logmsg('request:' + str(req))
    subid = req.json.get('subid')
    try:
        sub = Submission.objects.get(pk=subid)
    except Submission.DoesNotExist:
        return HttpResponseErrorJson("submission does not exist")
    image_ids = []
    for image in sub.user_images.all():
        image_ids += [image.id]
    return HttpResponseJson({'status': 'success',
                             'image_ids': image_ids})

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

@csrf_exempt
def job_status(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    status = job.get_status_blurb()
    return HttpResponseJson({
        'status':status,
    })

@csrf_exempt
def calibration(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    if job.calibration:
        (ra, dec, radius) = job.calibration.get_center_radecradius()
        return HttpResponseJson({
            'ra':ra,
            'dec':dec,
            'radius':radius,
        })
    else:
        return HttpResponseJson({
            'error':'no calibration data available for job %d' % int(job_id)
        })

@csrf_exempt
def tags(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    tags = job.user_image.tags.all()
    json_tags = [tag.text for tag in tags]
    return HttpResponseJson({
        'tags':json_tags}
    )
