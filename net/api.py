from __future__ import print_function
from __future__ import absolute_import
import base64
import os
from functools import wraps

if __name__ == '__main__':
    os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
    import sys
    fn = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(fn)
    import django
    django.setup()

from django.contrib import auth
from django.shortcuts import get_object_or_404
from django.http import HttpResponse, HttpResponseBadRequest
from django.core.exceptions import ObjectDoesNotExist
from django.views.decorators.csrf import csrf_exempt

# astrometry.net imports
from astrometry.net.models import *
from astrometry.net.views.submission import handle_file_upload
from astrometry.net.api_util import *
from astrometry.net.log import *
from astrometry.net.tmpfile import *
import astrometry.net.settings

# Content-type to return for JSON outputs.
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
        if not user.is_authenticated:
            return HttpResponseErrorJson('user is not authenticated.')
        return handler(request, *args, **kwargs)
    return handle_request


def upload_common(request, url=None, file=None):
    json = request.json
    logmsg('upload: JSON', json)
    df = None
    original_filename = ''
    # Handle X,Y coordinate lists
    if 'x' in json and 'y' in json:
        import numpy as np
        from astrometry.util.fits import fits_table
        # Turns out the easiest way to interface this with the rest of the code
        # is to just write a FITS table...
        try:
            x = np.array([float(v) for v in json['x']])
            y = np.array([float(v) for v in json['y']])
        except:
            return HttpResponseErrorJson('Failed to parse JSON "x" and "y" values -- should be lists of floats')
        if len(x) != len(y):
            return HttpResponseErrorJson('"x" and "y" lists must be the same length')
        T = fits_table()
        T.x = x
        T.y = y
        temp_file_path = tempfile.mktemp()
        T.writeto(temp_file_path)
        df = DiskFile.from_file(temp_file_path,
                                collection=Image.ORIG_COLLECTION)
        original_filename = 'json-xy'
    elif file is not None:
        df, original_filename = handle_file_upload(file, tempfiles=request.tempfiles)

    submittor = request.user if request.user.is_authenticated else None
    pro = get_user_profile(submittor)
    allow_commercial_use = json.get('allow_commercial_use', 'd')
    allow_modifications = json.get('allow_modifications', 'd')
    mylicense, _ = License.objects.get_or_create(
        default_license=pro.default_license,
        allow_commercial_use=allow_commercial_use,
        allow_modifications=allow_modifications,
    )
    publicly_visible = json.get('publicly_visible', 'y')

    subargs = dict(
        user=submittor,
        disk_file=df,
        original_filename=original_filename,
        license=mylicense,
        publicly_visible=publicly_visible,
        via_api = True,
        )
    if url is not None:
        subargs.update(url=url)

    if 'album' in json:
        albumname = str(json['album'])
        albums = Album.objects.filter(title=str(albumname))
        if albums.count() == 0:
            return HttpResponseErrorJson('Failed to find an Album with title "%s"' % albumname)
        album = albums[0]
        subargs.update(album=album)

    for key,typ in [('scale_units', str),
                    ('scale_type', str),
                    ('scale_lower', float),
                    ('scale_upper', float),
                    ('scale_est', float),
                    ('scale_err', float),
                    ('center_ra', float),
                    ('center_dec', float),
                    ('radius', float),
                    ('tweak_order', int),
                    ('downsample_factor', int),
                    ('use_sextractor', bool),
                    ('crpix_center', bool),
                    ('invert', bool),
                    ('parity', int),
                    ('image_width', int),
                    ('image_height', int),
                    ('positional_error', float),
                    ]:
        if key in json:
            subargs[key] = typ(json[key])

    logmsg('upload: submission args:', subargs)

    sub = Submission(**subargs)
    sub.save()

    rtn = {'status': 'success',
           'subid': sub.id, }
    if sub.disk_file is not None:
        rtn.update({'hash': sub.disk_file.file_hash})
    return HttpResponseJson(rtn)

@csrf_exempt
@requires_json_args
@requires_json_session
@requires_json_login
def url_upload(req):
    #logmsg('request:' + str(req))
    url = req.json.get('url')
    logmsg('url: %s' % url)

    logmsg('req.session:', req.session)
    logmsg('req.session keys:', req.session.keys())
    user_id = req.session['_auth_user_id']
    logmsg('user_id:', user_id)
    backend_path = req.session['_auth_user_backend']
    logmsg('backend_path:', backend_path)
    from django.contrib.auth import load_backend

    backend = load_backend(backend_path)
    logmsg('backend:', backend)
    user = backend.get_user(user_id)
    logmsg('user:', user)
    logmsg('is_auth:', user.is_authenticated)

    logmsg('request.user:', req.user)

    return upload_common(req, url=url)

@csrf_exempt
@requires_json_args
@requires_json_session
@requires_json_login
def api_upload(request):
    #logmsg('request:' + str(request))
    upfile = request.FILES.get('file', None)
    logmsg('api_upload: got file', upfile)
    logmsg('request.POST has keys:', request.POST.keys())
    logmsg('request.GET has keys:', request.GET.keys())
    logmsg('request.FILES has keys:', request.FILES.keys())
    #logmsg('api_upload: got request: ' + str(request.FILES['file'].size))
    return upload_common(request, file=request.FILES.get('file'))

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
    from .sdss_image import plot_sdss_image
    wcsfn = get_temp_file(tempfiles=req.tempfiles)
    plotfn = get_temp_file(tempfiles=req.tempfiles)
    write_wcs_file(req, wcsfn)
    plot_sdss_image(wcsfn, plotfn)
    res = HttpResponseJson({'status': 'success',
                             'plot': base64.b64encode(open(plotfn).read()),
                             })
    return res

@csrf_exempt
@requires_json_args
@requires_json_session
def api_galex_image_for_wcs(req):
    from .galex_jpegs import plot_into_wcs
    wcsfn = get_temp_file(tempfiles=req.tempfiles)
    plotfn = get_temp_file(tempfiles=req.tempfiles)
    write_wcs_file(req, wcsfn)
    plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR)
    res = HttpResponseJson({'status': 'success',
                             'plot': base64.b64encode(open(plotfn).read()),
                             })
    return res

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
        loginfo('Matched API key: ' + str(profile))
    except ObjectDoesNotExist:
        loginfo('Bad API key')
        return HttpResponseErrorJson('bad apikey')
    except:
        import traceback
        loginfo('Error matching API key: ' + traceback.format_exc())
        raise


    # Successful API login.  Register on the django side.

    # can't do this:
    #password = profile.user.password
    #auth.authenticate(username=profile.user.username, password=password)
    #auth.login(request, profile.user)

    # Instead, since we're using the auth AuthenticationMiddleware,
    # we just have to register the session?

    #loginfo('backends:' + str(auth.get_backends()))

    user = profile.user
    #user.backend = 'django_openid_auth.auth.OpenIDBackend'
    user.backend = 'django.contrib.auth.backends.ModelBackend'

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
@requires_json_args
@requires_json_session
@requires_json_login
def myjobs(request):
    jobs = Job.objects.filter(user_image__user=auth.get_user(request))
    return HttpResponseJson({ 'jobs': [j.id for j in jobs], 'status':'success'})

@csrf_exempt
def submission_status(req, sub_id):
    sub = get_object_or_404(Submission, pk=sub_id)
    jobs = []
    jobcals = []
    for job in sub.get_best_jobs():
        if job is None:
            jobs.append(None)
        else:
            jobs.append(job.id)
            cal = job.calibration
            if cal is not None:
                jobcals.append((job.id, cal.id))

    json_response = {
        'user':sub.user.id,
        'processing_started':str(sub.processing_started),
        'processing_finished':str(sub.processing_finished),
        'user_images':[uimage.id for uimage in sub.user_images.all()],
        'images':[uimage.image.id for uimage in sub.user_images.all()],
        'jobs':jobs,
        'job_calibrations':jobcals,
    }

    if sub.error_message:
        json_response.update({'error_message':sub.error_message})
    return HttpResponseJson(json_response)

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
        cal = job.calibration
        (ra, dec, radius) = cal.get_center_radecradius()
        pixscale = cal.raw_tan.get_pixscale()
        orient = cal.get_orientation()
        return HttpResponseJson({
            'ra':ra,
            'dec':dec,
            'width_arcsec': pixscale * cal.raw_tan.imagew,
            'height_arcsec': pixscale * cal.raw_tan.imageh,
            'radius':radius,
            'pixscale':pixscale,
            'orientation':orient,
            'parity': cal.get_parity(),
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

@csrf_exempt
def machine_tags(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    machine_user = User.objects.get(username=MACHINE_USERNAME)
    tags = TaggedUserImage.objects.filter(
        user_image = job.user_image,
        tagger = machine_user
    )
    json_tags = [tagged_user_image.tag.text for tagged_user_image in tags]
    return HttpResponseJson({
        'tags':json_tags}
    )

@csrf_exempt
def objects_in_field(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    sky_objects = job.user_image.sky_objects.all()
    json_sky_objects = [sky_obj.name for sky_obj in sky_objects]
    return HttpResponseJson({
        'objects_in_field':json_sky_objects}
    )

def get_anns(cal, nbright=0):
    wcsfn = cal.get_wcs_file()
    from astrometry.util.util import anwcs
    wcs = anwcs(wcsfn,0)
    catdir = settings.CAT_DIR
    uzcfn = os.path.join(catdir, 'uzc2000.fits')
    abellfn = os.path.join(catdir, 'abell-all.fits')
    ngcfn = os.path.join(catdir, 'openngc-ngc.fits')
    ngcnamesfn = os.path.join(catdir, 'openngc-names.fits')
    icfn = os.path.join(catdir, 'openngc-ic.fits')
    brightfn = os.path.join(catdir, 'brightstars.fits')
    hdfn = settings.HENRY_DRAPER_CAT
    hipfn = settings.HIPPARCOS_CAT
    tycho2fn = settings.TYCHO2_KD

    import astrometry.plot.plotann as plotann
    opt = plotann.get_empty_opts()
    if nbright:
        opt.nbright = nbright
    rad = cal.get_radius()
    # These are the same limits used in views/image.py for annotations
    if rad < 1.:
        if os.path.exists(abellfn):
            opt.abellcat = abellfn
        else:
            print('Abell catalog file does not exist:', abellfn)
        if os.path.exists(hdfn):
            opt.hdcat = hdfn
        else:
            print('Henry Draper catalog file does not exist:', hdfn)
    if rad < 0.25:
        if os.path.exists(tycho2fn):
            opt.t2cat = tycho2fn
        else:
            print('Tycho-2 catalog file does not exist:', tycho2fn)
        if os.path.exists(hipfn):
            opt.hipcat = hipfn
        else:
            print('Hipparcos catalog file does not exist:', hipfn)
    if rad < 10:
        if os.path.exists(ngcfn) and os.path.exists(ngcnamesfn) and os.path.exists(icfn):
            opt.ngc = True
            opt.ngccat = ngcfn
            opt.ngcname = ngcnamesfn
            opt.iccat = icfn
        else:
            print('NGC/IC catalog files do not exist:', ngcfn, ngcnamesfn, icfn)
    if os.path.exists(brightfn):
        opt.brightcat = brightfn
    else:
        print('Bright star catalog file does not exist:', brightfn)

    jobjs = plotann.get_annotations_for_wcs(wcs, opt)
    return jobjs

@csrf_exempt
def annotations_in_field(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    if not job.calibration:
        return HttpResponseJson({
            'error':'no calibration data available for job %d' % int(job_id)
        })
    cal = job.calibration
    kwa = {}
    if 'nbright' in req.GET:
        kwa.update(nbright=int(req.GET['nbright']))
    jobjs = get_anns(cal, **kwa)
    return HttpResponseJson({
        'annotations': jobjs})


@csrf_exempt
def job_info(req, job_id):
    job = get_object_or_404(Job, pk=job_id)
    ui = job.user_image
    sky_objects = ui.sky_objects.all()
    json_sky_objects = [sky_obj.name for sky_obj in sky_objects]

    tags = TaggedUserImage.objects.filter(user_image = ui)
    json_tags = [t.tag.text for t in tags]

    machine_user = User.objects.get(username=MACHINE_USERNAME)
    mtags = tags.filter(tagger = machine_user)
    machine_tags = [t.tag.text for t in mtags]

    status = job.get_status_blurb()

    result = {
        'objects_in_field':json_sky_objects,
        'machine_tags': machine_tags,
        'tags':json_tags,
        'status':status,
        'original_filename': ui.original_file_name,
        }

    if job.calibration:
        cal = job.calibration
        (ra, dec, radius) = cal.get_center_radecradius()
        pixscale = cal.raw_tan.get_pixscale()
        orient = cal.get_orientation()
        result['calibration'] = {
            'ra':ra,
            'dec':dec,
            'radius':radius,
            'pixscale':pixscale,
            'orientation':orient,
            'parity': cal.get_parity(),
            }


    return HttpResponseJson(result)

@csrf_exempt
def jobs_by_tag(req):
    query = req.GET.get('query')
    exact = req.GET.get('exact')
    images = UserImage.objects.all_visible()
    job_ids = []
    if exact:
        try:
            tag = Tag.objects.filter(text__iexact=query).get()
            images = images.filter(tags=tag)
            job_ids = [[job.id for job in image.jobs.all()] for image in images]
        except Tag.DoesNotExist:
            images = UserImage.objects.none()
    else:
        images = images.filter(tags__text__icontains=query)
        job_ids = [[job.id for job in image.jobs.all()] for image in images]

    # flatten job_ids list
    if job_ids:
        job_ids = [id for sublist in job_ids for id in sublist]

    return HttpResponseJson({
        'job_ids':job_ids}
    )




if __name__ == '__main__':
    #job = Job.objects.get(id=12)
    # job = Job.objects.get(id=1649169)
    # cal = job.calibration
    # print(get_anns(cal))
    # class Duck(object):
    #     pass
    # r = Duck()
    # print(submission_status(r, 2561176))

    import logging
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    from astrometry.net.models import Job
    job = Job.objects.get(id=7009522)
    skyobjs = job.calibration.get_objs_in_field()
    print('Sky objects:', skyobjs)

    ui = job.user_image

    print('UI tags:', ui.tags.all())
    print('UI sky objects:', ui.sky_objects.all())

    for t in ui.tags.all():
        print('Tag', t)
        tui = TaggedUserImage.objects.filter(tag=t, user_image=ui)
        print('TaggedUserImage:', tui)
        tui.delete()
    for t in ui.sky_objects.all():
        print('SkyObj', t)
        ui.sky_objects.remove(t)
    # Noooo
    #ui.tags.all().delete()
    #ui.sky_objects.all().delete()
    
    ui.add_machine_tags(job)
    ui.add_sky_objects(job)

    print('UI tags:', ui.tags.all())
    print('UI sky objects:', ui.sky_objects.all())

    import sys
    sys.exit(0)
    
    from django.test import Client
    c = Client()
    # anonymous
    #r = c.post('/api/login', data={'request-json': '{"apikey": "1jcrmadfnxngxscd"}'})
    r = c.get('/api/jobs/7009522/objects_in_field')
    print('Got response:', r)

    f = open('out', 'wb')
    for x in r:
        f.write(x)
    f.close()
    
