import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms
from django.contrib import messages

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from astrometry.net.tmpfile import *
from sdss_image import plot_sdss_image

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command

from astrometry.net.views.comment import *
from astrometry.net.util import get_page, get_session_form
from astrometry.net.views.tag import TagForm

from string import strip

def user_image(req, user_image_id=None):
    image = get_object_or_404(UserImage, pk=user_image_id)

    job = image.get_best_job() 
    calib = None
    if job:
        calib = job.calibration

    comment_form = get_session_form(req.session, PartialCommentForm)
    tag_form = get_session_form(req.session, TagForm)

    logmsg(image.get_absolute_url())
    context = {
        'display_image': image.image.get_display_image(),
        'image': image,
        'job': job,
        'calib': calib,
        'comment_form': comment_form,
        'tag_form': tag_form,
    }

    if image.is_public() or (image.user == req.user and req.user.is_authenticated()):
        template = 'user_image/view.html'
    elif SharedHideable.objects.filter(shared_with=req.user.id, hideable=image).count():
        template = 'user_image/view.html'
    else:
        messages.error(req, "Sorry, you don't have permission to view this content")
        template = 'user_image/permission_denied.html'
    return render_to_response(template, context,
        context_instance = RequestContext(req))

def serve_image(req, id=None):
    image = get_object_or_404(Image, pk=id)
    res = HttpResponse(mimetype=image.get_mime_type())
    image.render(res)
    return res

def annotated_image(req, jobid=None, size='full'):
    job = get_object_or_404(Job, pk=jobid)
    ui = job.user_image
    img = ui.image
    if size == 'display':
        scale = float(img.get_display_image().width)/img.width
        img = img.get_display_image()
    else:
        scale = 1.0
        
    if hasattr(img, 'sourcelist'):
        imgfn = get_temp_file()
        f = open(imgfn,'wb')
        img.render(f)
        f.close()
    else:
        df = img.disk_file
        imgfn = df.get_path()
    wcsfn = job.get_wcs_file()
    pnmfn = get_temp_file()
    (filetype, errstr) = image2pnm.image2pnm(imgfn, pnmfn)
    if errstr:
        logmsg('Error converting image file %s: %s' % (imgfn, errstr))
        return HttpResponse('plot failed')
    annfn = get_temp_file()
    cmd = 'plot-constellations -w %s -i %s -o %s -s %s -N -C -B' % (wcsfn, pnmfn, annfn, str(scale))
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('plot failed')
    f = open(annfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

def onthesky_image(req, zoom=None, calid=None):
    from astrometry.net.views.onthesky import plot_aitoff_wcs_outline
    from astrometry.net.views.onthesky import plot_wcs_outline
    from astrometry.util import util as anutil
    #
    cal = get_object_or_404(Calibration, pk=calid)
    wcsfn = cal.get_wcs_file()
    plotfn = get_temp_file()
    #
    wcs = anutil.Tan(wcsfn, 0)
    zoom = int(zoom)
    if zoom == 0:
        zoom = wcs.radius() < 15.
        plot_aitoff_wcs_outline(wcsfn, plotfn, zoom=zoom)
    elif zoom == 1:
        zoom = wcs.radius() < 1.5
        plot_wcs_outline(wcsfn, plotfn, zoom=zoom)
    elif zoom == 2:
        zoom = wcs.radius() < 0.15
        plot_wcs_outline(wcsfn, plotfn, width=3.6, grid=1, zoom=zoom,
                         zoomwidth=0.36)
        # hd=True is too cluttered at this level
    elif zoom == 3:
        plot_wcs_outline(wcsfn, plotfn, width=0.36, grid=0.1, zoom=False,
                         hd=False)
    else:
        return HttpResponse('invalid zoom')
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res


def galex_image(req, calid=None, size='full'):
    from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    from astrometry.net.galex_jpegs import plot_into_wcs
    cal = get_object_or_404(Calibration, pk=calid)
    key = 'galex_size%s_cal%i' % (size, cal.id)
    df = CachedFile.get(key)
    if df is None:
        wcsfn = cal.get_wcs_file()
        plotfn = get_temp_file()
        if size == 'display':
            image = cal.jobs.get().user_image
            scale = float(image.image.get_display_image().width)/image.image.width
        else:
            scale = 1.0
        plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR, scale=scale)
        # cache
        logmsg('Caching key "%s"' % key)
        df = CachedFile.add(key, plotfn)
    else:
        logmsg('Cache hit for key "%s"' % key)
    f = open(df.get_path())
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res


def sdss_image(req, calid=None, size='full'):
    cal = get_object_or_404(Calibration, pk=calid)
    key = 'sdss_size%s_cal%i' % (size, cal.id)
    df = CachedFile.get(key)
    if df is None:
        wcsfn = cal.get_wcs_file()
        plotfn = get_temp_file()
        if size == 'display':
            image = cal.jobs.get().user_image
            scale = float(image.image.get_display_image().width)/image.image.width
        else:
            scale = 1.0
        plot_sdss_image(wcsfn, plotfn, scale)
        # cache
        logmsg('Caching key "%s"' % key)
        df = CachedFile.add(key, plotfn)
    else:
        logmsg('Cache hit for key "%s"' % key)
    f = open(df.get_path())
    res = HttpResponse(f)
    res['Content-type'] = 'image/png'
    return res

# 2MASS:
# Has a SIA service, but does not make mosaics.
# Documented here:
# http://irsa.ipac.caltech.edu/applications/2MASS/IM/docs/siahelp.html
# Eg:
# http://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia?POS=187.03125,16.9577633&SIZE=1&type=ql

# WISE:
# -IPAC/IRSA claims to have VO services
# -elaborate javascripty interface

def index(req, images=UserImage.objects.all().order_by('-submission__submitted_on')[:9], 
            template_name='user_image/index_recent.html', context={}):

    page_number = req.GET.get('page',1)
    page = get_page(images,3*10,page_number)
    context.update({'image_page':page})
    return render_to_response(template_name,   
        context,
        context_instance = RequestContext(req))


def index_recent(req):
    return index(req, 
                 UserImage.objects.all().order_by('-submission__submitted_on')[:9],
                 template_name='user_image/index_recent.html')

def index_all(req):
    return index(req,
                 UserImage.objects.all().order_by('-submission__submitted_on'),
                 template_name='user_image/index_all.html')

def index_user(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)
    return index(req,
                 user.user_images.all().order_by('-submission__submitted_on'),
                 template_name='user_image/index_user.html',
                 context={'display_user':user})

def index_by_user(req):
    # make ordering case insensitive
    context = {
        'users':User.objects.all().order_by('profile__display_name', 'id')
    }
    return render_to_response('user_image/index_by_user.html',
        context,
        context_instance = RequestContext(req))
        
def index_album(req, album_id=None):
    album = get_object_or_404(Album, pk=album_id)
    return index(req,
                 album.user_images.all(),
                 template_name='user_image/index_album.html',
                 context={'album':album})

def image_set(req, category, id):
    default_category = 'user'
    cat_classes = {
        'user':User,
        'album':Album,
        'tag':Tag,
    }

    if category not in cat_classes:
        category = default_category

    cat_class = cat_classes[category]
    cat_obj = get_object_or_404(cat_class, pk=id)
    
    set_names = {
        'user':'Submitted by User %s' % cat_obj.pk,
        'album':'Album: %s' % cat_obj.pk,
        'tag':'Tag: %s' % cat_obj.pk,
    } 
    image_set_title = set_names[category]

    context = {
        'images': cat_obj.user_images.all,
        'image_set_title':image_set_title,
    }
   
    return render_to_response('user_image/image_set.html',
        context,
        context_instance = RequestContext(req))

def wcs_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_wcs_file())
    res = HttpResponse(f)
    res['Content-type'] = 'application/fits' 
    res['Content-Disposition'] = 'attachment; filename=wcs.fits'
    return res

class ImageSearchForm(forms.Form):
    SEARCH_CATEGORIES = (('tag', 'By Tag'),
                         ('user', 'By User'),
                         ('location', 'By Location'))

    search_category = forms.ChoiceField(widget=forms.HiddenInput(),
                                        choices=SEARCH_CATEGORIES,
                                        initial='tag',
                                        required=False)

    tags = forms.CharField(required=False)
    user = forms.CharField(required=False)
    calibrated_only = forms.BooleanField(initial=False,required=False)
    
    def clean(self):
        category = self.cleaned_data.get('search_category');
        if not category:
            self.cleaned_data['search_category'] = 'tag'

        return self.cleaned_data
    
def unhide(req, user_image_id):
    image = get_object_or_404(UserImage, pk=user_image_id)
    if req.user.is_authenticated and req.user == image.user:
        image.unhide()
    return redirect('astrometry.net.views.image.user_image', user_image_id)
    
def hide(req, user_image_id):
    image = get_object_or_404(UserImage, pk=user_image_id)
    if req.user.is_authenticated and req.user == image.user:
        image.hide()
    return redirect('astrometry.net.views.image.user_image', user_image_id)
    
def search(req):
    form = ImageSearchForm(req.GET)

    context = {}
    page = None
    if form.is_valid(): 
        all_images = UserImage.objects.all()
        images = all_images
        category = form.cleaned_data.get('search_category');
        if category == 'tag':
            tags = form.cleaned_data.get('tags','')
            if tags.strip():
                images = UserImage.objects.none()
                tag_objs = []
                tags = map(strip,tags.split(','))
                tags = list(set(tags)) # remove duplicate tags
                
                images = all_images.filter(tags__text__in=tags).distinct()
                tag_objs = Tag.objects.filter(text__in=tags)
                context['tags'] = tag_objs

        elif category == 'user':
            username = form.cleaned_data.get('user','')

            if username.strip():
                user = User.objects.filter(profile__display_name=username)[:1]
                images = UserImage.objects.none()
                if len(user) > 0:
                    images = all_images.filter(user=user)
                    context['display_user'] = user[0] 
                else:
                    context['display_users'] = User.objects.filter(profile__display_name__startswith=username)[:5]
        
        calibrated_only = form.cleaned_data.get('calibrated_only')
        if calibrated_only:
            images = images.filter(jobs__calibration__isnull=False)
        page_number = req.GET.get('page',1)
        page = get_page(images.order_by('-submission__submitted_on'),4*5,page_number)

    context.update({'form': form,
                    'search_category': form.cleaned_data.get('search_category'),
                    'image_page': page})
    return render_to_response('user_image/search.html',
        context,
        context_instance = RequestContext(req))
