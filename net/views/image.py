import shutil
import os, errno
import hashlib
import tempfile
import math
import urllib
import urllib2
import PIL.Image
import stat
import time
from datetime import datetime, timedelta

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect, render
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms
from django.contrib import messages

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from astrometry.net.tmpfile import *
from sdss_image import plot_sdss_image

from astrometry.blind.plotstuff import *
from astrometry.util import image2pnm
from astrometry.util.run_command import run_command
from astrometry.util.file import *

from astrometry.net.models import License

from astrometry.net.views.comment import *
from astrometry.net.views.license import *
from astrometry.net.util import get_page, get_session_form, NoBulletsRenderer
from astrometry.net.views.tag import TagForm, TagSearchForm
from astrometry.net.views.license import LicenseForm

from string import strip
import simplejson

class UserImageForm(forms.ModelForm):
    album = forms.ChoiceField(choices=(), required=False)
    class Meta:
        model = UserImage
        exclude = (
            'image',
            'user',
            'tags',
            'flags',
            'original_file_name',
            'submission',
            'owner',
            'comment_receiver',
            'license',
            'sky_objects',
        )
        widgets = {
            'description': forms.Textarea(attrs={'cols':60,'rows':3}),
            'publicly_visible': forms.RadioSelect(renderer=NoBulletsRenderer),
        }

    def __init__(self, user, *args, **kwargs):
        super(UserImageForm, self).__init__(*args, **kwargs)
        self.fields['album'].choices = [('', 'none')]
        self.fields['album'].initial = ''
        user_image = kwargs.get('instance')
        if user.is_authenticated():
            for album in Album.objects.filter(user=user).all():
                self.fields['album'].choices += [(album.id, album.title)]
                if user_image and user_image in album.user_images.all():
                    self.fields['album'].initial = album.id

def user_image(req, user_image_id=None):
    uimage = get_object_or_404(UserImage, pk=user_image_id)

    job = uimage.get_best_job() 
    calib = None
    if job:
        calib = job.calibration

    #license_form = get_session_form(req.session, PartialLicenseForm)
    comment_form = get_session_form(req.session, PartialCommentForm)
    tag_form = get_session_form(req.session, TagForm)

    images = {}
    dim = uimage.image.get_display_image()
    images['original_display'] = reverse('astrometry.net.views.image.serve_image', kwargs={'id':dim.id})
    images['original'] = reverse('astrometry.net.views.image.serve_image', kwargs={'id':uimage.image.id})
    image_type = 'original'
    if job:
        if job.calibration:
            images['annotated_display'] = reverse('annotated_image', kwargs={'jobid':job.id,'size':'display'})
            images['annotated'] = reverse('annotated_image', kwargs={'jobid':job.id,'size':'full'})
            images['sdss_display'] = reverse('sdss_image', kwargs={'calid':job.calibration.id,'size':'display'})
            images['sdss'] = reverse('sdss_image', kwargs={'calid':job.calibration.id,'size':'full'})
            images['galex_display'] = reverse('galex_image', kwargs={'calid':job.calibration.id,'size':'display'})
            images['galex'] = reverse('galex_image', kwargs={'calid':job.calibration.id,'size':'full'})
            images['redgreen_display'] = reverse('red_green_image', kwargs={'job_id':job.id,'size':'display'})
            images['redgreen'] = reverse('red_green_image', kwargs={'job_id':job.id,'size':'full'})
            image_type = 'annotated'
        images['extraction_display'] = reverse('astrometry.net.views.image.extraction_image', kwargs={'job_id':job.id,'size':'display'})
        images['extraction'] = reverse('astrometry.net.views.image.extraction_image', kwargs={'job_id':job.id,'size':'full'})

    image_type = req.GET.get('image', image_type)
    if image_type in images:
        display_url = images[image_type + '_display']
        fullsize_url = images[image_type]

    flags = Flag.objects.all()
    if req.user.is_authenticated():
        selected_flags = [flagged_ui.flag for flagged_ui in
            FlaggedUserImage.objects.filter(
                user_image=uimage,
                user=req.user,
            )
        ]
    else:
        selected_flags = None
        
    logmsg(uimage.get_absolute_url())
    context = {
        'request': req,
        'display_image': dim,
        'image': uimage,
        'job': job,
        'calib': calib,
        'comment_form': comment_form,
        #'license_form': license_form,
        'tag_form': tag_form,
        'images': simplejson.dumps(images),
        'display_url': display_url,
        'fullsize_url': fullsize_url,
        'image_type': image_type,
        'flags': flags,
        'selected_flags': selected_flags,
    }

    if uimage.is_public() or (req.user.is_authenticated() and uimage.user == req.user):
        template = 'user_image/view.html'
    #elif SharedHideable.objects.filter(shared_with=req.user.id, hideable=image).count():
    #    template = 'user_image/view.html'
    else:
        messages.error(req, "Sorry, you don't have permission to view this content.")
        template = 'user_image/permission_denied.html'
    return render_to_response(template, context,
        context_instance = RequestContext(req))

@login_required
def edit(req, user_image_id=None):
    user_image = get_object_or_404(UserImage, pk=user_image_id) 
    if user_image.user != req.user:
        messages.error(req, "Sorry, you don't have permission to edit this content.")
        return render(req, 'user_image/permission_denied.html')

    if req.method == 'POST':
        image_form = UserImageForm(req.user, req.POST, instance=user_image)
        license_form = LicenseForm(req.POST)
        if image_form.is_valid() and license_form.is_valid():
            image_form.save()

            license,created = License.objects.get_or_create(
                default_license=req.user.get_profile().default_license,
                allow_commercial_use=license_form.cleaned_data['allow_commercial_use'],
                allow_modifications=license_form.cleaned_data['allow_modifications'],
            )
            user_image.license = license
            
            album_id = image_form.cleaned_data['album']
            albums = Album.objects.filter(user=req.user).filter(user_images__in=[user_image])
            if album_id == '':
                # remove user_image from album
                for album in albums:
                    album.user_images.remove(user_image)
            else:
                try:
                    album = Album.objects.get(pk=int(album_id))
                    if album not in albums:
                        # move user_image to new album
                        for album in albums:
                            album.user_images.remove(user_image)
                        album.user_images.add(user_image)
                except Exception as e:
                    pass

            selected_flags = req.POST.getlist('flags')
            user_image.update_flags(selected_flags, req.user)
            user_image.save()

            messages.success(req, 'Image details successfully updated.')
            return redirect(user_image)
        else:
            messages.error(req, 'Please fix the following errors:')
    else:
        image_form = UserImageForm(req.user, instance=user_image)
        license_form = LicenseForm(instance=user_image.license)
        flags = Flag.objects.all()
        selected_flags = [flagged_ui.flag for flagged_ui in
            FlaggedUserImage.objects.filter(
                user_image=user_image,
                user=req.user,
            )
        ]
        
    context = {
        'image_form': image_form,
        'license_form': license_form,
        'flags': flags,
        'selected_flags':selected_flags,
        'image': user_image,
    }
    return render(req, 'user_image/edit.html', context)

def serve_image(req, id=None):
    image = get_object_or_404(Image, pk=id)
    res = HttpResponse(mimetype=image.get_mime_type())

    date = datetime.now() + timedelta(days=7)
    res['Expires'] = time.asctime(date.timetuple())
    mtime = os.stat(image.disk_file.get_path())[stat.ST_MTIME]
    res['Last-Modified'] = time.asctime(time.gmtime(mtime))
    if 'filename' in req.GET:
        res['Content-Disposition'] = 'filename=%s' % req.GET['filename']
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
        
    wcsfn = job.get_wcs_file()
    pnmfn = img.get_pnm_path()
    annfn = get_temp_file()

    #datadir = os.path.join(os.path.dirname(os.path.dirname(settings.WEB_DIR)), 'data')
    catdir = settings.CAT_DIR
    uzcfn = os.path.join(catdir, 'uzc2000.fits')
    abellfn = os.path.join(catdir, 'abell-all.fits')

    #hdfn = os.path.join(os.path.dirname(os.path.dirname(settings.WEB_DIR)),
    #'net', 'hd.fits')

    hdfn = settings.HENRY_DRAPER_CAT

    tycho2fn = settings.TYCHO2_KD

    rad = job.calibration.get_radius()

    #logmsg('pnm file: %s' % pnmfn)

    args = ['plotann.py --no-grid --toy -10',
            '--scale %s' % (str(scale)),]
    #if rad < 10.:
    if rad < 1.:
        args.extend([#'--uzccat %s' % uzcfn,
                     '--abellcat %s' % abellfn,
                     '--hdcat %s' % hdfn
                     ])

    if rad < 0.25:
        args.append('--tycho2cat %s' % tycho2fn)

    #if rad > 20:
    if rad > 10:
        args.append('--no-ngc')

	if rad > 30:
		args.append('--no-bright')
            
    cmd = ' '.join(args + ['%s %s %s' % (wcsfn, pnmfn, annfn)])

    #cmd = 'plot-constellations -w %s -i %s -o %s -s %s -N -C -B -c' % (wcsfn, pnmfn, annfn, str(scale))

    import sys

    # (rtn,out,err) = run_command('which plotann.py; echo pyp $PYTHONPATH; echo path $PATH; echo llp $LD_LIBRARY_PATH; echo "set"; set')
    # return HttpResponse('which: ' + out + err + '<br>sys.path<br>' + '<br>'.join(sys.path) +
    #                     "<br>PATH " + os.environ['PATH'] +
    #                     "<br>LLP " + os.environ['LD_LIBRARY_PATH'] +
    #                     "<br>sys.path " + ':'.join(sys.path) +
    #                     "<br>cmd " + cmd)

    os.environ['PYTHONPATH'] = ':'.join(sys.path)

    logmsg('Running: ' + cmd)
    #logmsg('PYTHONPATH: ' + os.environ['PYTHONPATH'])
    #logmsg('PATH: ' + os.environ['PATH'])
    #(rtn,out,err) = run_command('which plotann.py')
    #logmsg('which plotann.py: ' + out)

    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('plot failed: ' + err + "<br><pre>" + out + "</pre><br><pre>" + err + "</pre>")
    f = open(annfn)
    res = HttpResponse(f)
    #res['Content-Type'] = 'image/png'
    # plotann.py produces jpeg by default
    res['Content-Type'] = 'image/jpeg'
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
                         zoomwidth=0.36, hd=True, hd_labels=False,
                         tycho2=False)
        # hd=True is too cluttered at this level
    elif zoom == 3:
        plot_wcs_outline(wcsfn, plotfn, width=0.36, grid=0.1, zoom=False,
                         hd=True, hd_labels=True, tycho2=True)
    else:
        return HttpResponse('invalid zoom')
    f = open(plotfn)
    res = HttpResponse(f)
    res['Content-Type'] = 'image/png'
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
            image = cal.job.user_image
            scale = float(image.image.get_display_image().width)/image.image.width
        else:
            scale = 1.0

        # logmsg('WCS filename:', wcsfn)
        # logmsg('Plot filename:', plotfn)
        # logmsg('Basedir:', settings.GALEX_JPEG_DIR)
        # logmsg('Scale:', scale)

        plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR, scale=scale)
        # cache
        logmsg('Caching key "%s"' % key)
        df = CachedFile.add(key, plotfn)
    else:
        logmsg('Cache hit for key "%s"' % key)
    f = open(df.get_path())
    res = HttpResponse(f)
    res['Content-Type'] = 'image/png'
    return res


def sdss_image(req, calid=None, size='full'):
    cal = get_object_or_404(Calibration, pk=calid)
    key = 'sdss_size%s_cal%i' % (size, cal.id)
    df = CachedFile.get(key)
    if df is None:
        wcsfn = cal.get_wcs_file()
        plotfn = get_temp_file()
        if size == 'display':
            image = cal.job.user_image
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
    res['Content-Type'] = 'image/png'
    return res

def red_green_image(req, job_id=None, size='full'):
    job = get_object_or_404(Job, pk=job_id)
    ui = job.user_image
    img = ui.image
    if size == 'display':
        scale = float(img.get_display_image().width)/img.width
        img = img.get_display_image()
    else:
        scale = 1.0
        
    axyfn = job.get_axy_file()
    wcsfn = job.get_wcs_file()
    rdlsfn = job.get_rdls_file()
    pnmfn = img.get_pnm_path()
    exfn = get_temp_file()

    try:
        plot = Plotstuff()
        plot.wcs_file = wcsfn
        plot.outformat = PLOTSTUFF_FORMAT_PNG
        plot.outfn = exfn
        plot.scale_wcs(scale)
        plotstuff_set_size_wcs(plot.pargs)

        # plot image
        pimg = plot.image
        pimg.set_file(str(pnmfn))
        pimg.format = PLOTSTUFF_FORMAT_PPM
        plot.plot('image')

        # plot red
        xy = plot.xy
        if hasattr(img, 'sourcelist'):
            # set xy offsets for source lists
            fits = img.sourcelist.get_fits_table()
            #xy.xoff = int(fits.x.min())
            #xy.yoff = int(fits.y.min())
            xy.xoff = 0.
            xy.yoff = 0.
            
        plot_xy_set_filename(xy, str(axyfn))
        xy.scale = scale
        plot.color = 'red'
        xy.nobjs = 200
        plot.lw = 2.
        plot.markersize = 6
        plot.plot('xy')
        
        # plot green 
        rd = plot.radec
        plot_radec_set_filename(rd, str(rdlsfn))
        plot.color = 'green'
        plot.markersize = 4
        plot.plot('radec')

        plot.write()
    except:
        return HttpResponse("plot failed") 

    f = open(exfn)
    res = HttpResponse(f)
    res['Content-Type'] = 'image/png'
    return res

def extraction_image(req, job_id=None, size='full'):
    job = get_object_or_404(Job, pk=job_id)
    ui = job.user_image
    sub = ui.submission
    img = ui.image
    if size == 'display':
        scale = float(img.get_display_image().width)/img.width
        img = img.get_display_image()
    else:
        scale = 1.0
        
    axyfn = job.get_axy_file()
    pnmfn = img.get_pnm_path()
    exfn = get_temp_file()

    try:
        plot = Plotstuff()
        plot.size = [img.width, img.height]
        plot.outformat = PLOTSTUFF_FORMAT_PNG
        plot.outfn = exfn

        # plot image
        pimg = plot.image
        pimg.set_file(str(pnmfn))
        pimg.format = PLOTSTUFF_FORMAT_PPM
        plot.plot('image')

        # plot sources
        xy = plot.xy
        if hasattr(img, 'sourcelist'):
            # set xy offsets for source lists
            fits = img.sourcelist.get_fits_table()
            #xy.xoff = int(fits.x.min())
            #xy.yoff = int(fits.y.min())
            xy.xoff = xy.yoff = 1.

        if sub.use_sextractor:
            xy.xcol = 'X_IMAGE'
            xy.ycol = 'Y_IMAGE'

        plot_xy_set_filename(xy, str(axyfn))
        xy.scale = scale
        plot.color = 'red'
        # plot 50 brightest
        xy.firstobj = 0
        xy.nobjs = 50
        plot.lw = 2.
        plot.markersize = 6
        plot.plot('xy')
        # plot 200 other next brightest sources
        xy.firstobj = 50
        xy.nobjs = 250
        plot.alpha = 0.9
        plot.lw = 1.
        plot.markersize = 4
        plot.plot('xy')
        # plot 250 other next brightest sources
        xy.firstobj = 250
        xy.nobjs = 500
        plot.alpha = 0.5
        plot.lw = 1.
        plot.markersize = 3
        plot.plot('xy')
        plot.write()
    except:
        return HttpResponse("plot failed") 

    f = open(exfn)
    res = HttpResponse(f)
    res['Content-Type'] = 'image/png'
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

class ShowImagesForm(forms.Form):
    calibrated = forms.BooleanField(widget=forms.CheckboxInput(
                                        attrs={'onClick':'this.form.submit();'}),
                                    initial=True, required=False)
    processing = forms.BooleanField(widget=forms.CheckboxInput(
                                        attrs={'onClick':'this.form.submit();'}),
                                    initial=False, required=False)
    failed = forms.BooleanField(widget=forms.CheckboxInput(
                                        attrs={'onClick':'this.form.submit();'}),
                                    initial=False, required=False)

def index(req, images=None, 
          template_name='user_image/index.html', context={}):
    if images is None:
        images = UserImage.objects.public_only(req.user)
    form_data = req.GET.copy()
    if not (req.GET.get('calibrated')
            or req.GET.get('processing')
            or req.GET.get('failed')):
        form_data['calibrated'] = 'on'
    form = ShowImagesForm(form_data)
    calibrated = True
    processing = False
    failed = False
    if form.is_valid():
        calibrated = form.cleaned_data.get('calibrated')
        processing = form.cleaned_data.get('processing')
        failed = form.cleaned_data.get('failed')
        
    if calibrated is False:
        images = images.exclude(jobs__status='S')
    if processing is False:
        images = images.exclude(jobs__status='')
    if failed is False:
        images = images.exclude(jobs__status='F')

    images = images.order_by('-submission__submitted_on')
    page_number = req.GET.get('page', 1)
    page = get_page(images, 27, page_number)
    context.update({
        'image_page': page,
        'show_images_form': form,
    })
    return render(req, template_name, context)

def index_tag(req):
    images = UserImage.objects.public_only(req.user)
    form = TagSearchForm(req.GET)
    tag = None
    if form.is_valid():
        query = form.cleaned_data.get('query')
        exact = form.cleaned_data.get('exact')
        if query:
            if exact:
                try:
                    tag = Tag.objects.filter(text__iexact=query).get()
                    images = images.filter(tags=tag)
                except Tag.DoesNotExist:
                    images = UserImage.objects.none() 
            else:
                images = images.filter(tags__text__icontains=query)

    images = images.distinct()

    context = {
        'tag_search_form': form,
        'tag': tag,
    }
    return index(req, images, 'user_image/index_tag.html', context)
   
class LocationSearchForm(forms.Form):
    ra = forms.FloatField(widget=forms.TextInput(attrs={'size':'5'}))
    dec = forms.FloatField(widget=forms.TextInput(attrs={'size':'5'}))
    radius = forms.FloatField(widget=forms.TextInput(attrs={'size':'5'}))

def index_location(req):
    images = UserImage.objects.public_only(req.user)
    form = LocationSearchForm(req.GET)
    if form.is_valid():
        ra = form.cleaned_data.get('ra', 0)
        dec = form.cleaned_data.get('dec', 0)
        radius = form.cleaned_data.get('radius', 0)

        if ra and dec and radius: 
            ra *= math.pi/180
            dec *= math.pi/180
            tempr = math.cos(dec)
            x = tempr*math.cos(ra)
            y = tempr*math.sin(ra)
            z = math.sin(dec)
            r = radius/180*math.pi
           
            # HACK - there's probably a better way to do this..?
            where = ('(x-(%(x)f))*(x-(%(x)f))+(y-(%(y)f))*(y-(%(y)f))+(z-(%(z)f))*(z-(%(z)f)) < (%(r)f)*(%(r)f)'
                    % dict(x=x,y=y,z=z,r=r))
            where2 = '(r <= %f)' % r
            cals = Calibration.objects.extra(where=[where,where2])
            images = images.filter(jobs__calibration__in=cals)

    images = images.distinct()
    context = {
        'location_search_form': form,
    }
    return index(req, images, 'user_image/index_location.html', context)

def index_nearby(req, user_image_id=None):
    image = get_object_or_404(UserImage, pk=user_image_id)
    images = image.get_neighbouring_user_images()

    context = {
        'image': image,
    }
    return index(req, images, 'user_image/index_nearby.html', context)
    
def index_recent(req):
    return index(req, 
                 UserImage.objects.all_visible()[:9], #.order_by('-submission__submitted_on')[:9],
                 template_name='user_image/index_recent.html')

def index_all(req):
    return index(req,
                 UserImage.objects.all_visible().order_by('-submission__submitted_on'),
                 template_name='user_image/index_all.html')

def index_user(req, user_id=None):
    user = get_object_or_404(User, pk=user_id)
    return index(req,
                 user.user_images.all_visible().order_by('-submission__submitted_on'),
                 template_name='user_image/index_user.html',
                 context={'display_user':user})

def index_by_user(req):
    # make ordering case insensitive
    context = {
        'users':User.objects.all_visible().order_by('profile__display_name', 'id')
    }
    return render_to_response('user_image/index_by_user.html',
        context,
        context_instance = RequestContext(req))
        
def index_album(req, album_id=None):
    album = get_object_or_404(Album, pk=album_id)
    return index(req,
                 album.user_images.all_visible(),
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
    res['Content-Type'] = 'application/fits' 
    res['Content-Disposition'] = 'attachment; filename=wcs.fits'
    return res

def rdls_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_rdls_file())
    res = HttpResponse(f)
    res['Content-Type'] = 'application/fits' 
    res['Content-Disposition'] = 'attachment; filename=rdls.fits'
    return res

def axy_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_axy_file())
    res = HttpResponse(f)
    res['Content-Type'] = 'application/fits' 
    res['Content-Disposition'] = 'attachment; filename=axy.fits'
    return res

def corr_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_corr_file())
    res = HttpResponse(f)
    res['Content-Type'] = 'application/fits' 
    res['Content-Disposition'] = 'attachment; filename=corr.fits'
    return res

def new_fits_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    wcsfn = job.get_wcs_file()
    img = job.user_image.image
    df = img.disk_file
    infn = df.get_path()
    if df.is_fits_image():
        fitsinfn = infn
    else:
        ## FIXME -- could convert other formats to FITS...
        pnmfn = get_temp_file()
        fitsinfn = get_temp_file()
        cmd = 'image2pnm.py -i %s -o %s && an-pnmtofits %s > %s' % (infn, pnmfn, pnmfn, fitsinfn)
        logmsg('Running: ' + cmd)
        (rtn, out, err) = run_command(cmd)
        if rtn:
            logmsg('out: ' + out)
            logmsg('err: ' + err)
            return HttpResponse('image2pnm.py failed: out ' + out + ', err ' + err)
    outfn = get_temp_file()
    cmd = 'new-wcs -i %s -w %s -o %s -d' % (fitsinfn, wcsfn, outfn)
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('plot failed: out ' + out + ', err ' + err)
    res = HttpResponse(open(outfn))
    res['Content-Type'] = 'application/fits' 
    res['Content-Length'] = file_size(outfn)
    res['Content-Disposition'] = 'attachment; filename=new-image.fits'
    return res

def kml_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    wcsfn = job.get_wcs_file()
    img = job.user_image.image
    df = img.disk_file
   
    pnmfn = img.get_pnm_path()
    imgfn = get_temp_file()
    image = PIL.Image.open(pnmfn)
    image.save(imgfn, 'PNG') 

    dirnm = tempfile.mkdtemp()
    warpedimgfn = 'image.png'
    kmlfn = 'doc.kml'
    outfn = get_temp_file()
    cmd = ('cd %(dirnm)s'
           '; /usr/local/wcs2kml/bin/wcs2kml ' 
           '--input_image_origin_is_upper_left '
           '--fitsfile=%(wcsfn)s '
           '--imagefile=%(imgfn)s '
           '--kmlfile=%(kmlfn)s '
           '--outfile=%(warpedimgfn)s '
           '; zip -j - %(warpedimgfn)s %(kmlfn)s > %(outfn)s ' %
           dict(dirnm=dirnm, wcsfn=wcsfn, imgfn=imgfn, kmlfn=kmlfn, 
                warpedimgfn=warpedimgfn, outfn=outfn))
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('kml generation failed: ' + err)

    res = HttpResponse(open(outfn))
    res['Content-Type'] = 'application/x-zip-compressed'
    res['Content-Length'] = file_size(outfn)
    res['Content-Disposition'] = 'attachment; filename=image.kmz'
    return res

class ImageSearchForm(forms.Form):
    SEARCH_CATEGORIES = (('tag', 'By Tag'),
                         ('user', 'By User'),
                         ('location', 'By Location'),
                         ('image', 'By Image'))

    search_category = forms.ChoiceField(widget=forms.HiddenInput(),
                                        choices=SEARCH_CATEGORIES,
                                        initial='tag',
                                        required=False)
    tags = forms.CharField(widget=forms.TextInput(attrs={'autocomplete':'off'}),
                                                  required=False)
    user = forms.CharField(widget=forms.TextInput(attrs={'autocomplete':'off'}),
                                                  required=False)
    image = forms.IntegerField(widget=forms.HiddenInput(), required=False)

    ra = forms.FloatField(widget=forms.TextInput(attrs={'size':'5'}),required=False)
    dec = forms.FloatField(widget=forms.TextInput(attrs={'size':'5'}),required=False)
    radius = forms.FloatField(widget=forms.TextInput(attrs={'size':'5'}),required=False)

    calibrated = forms.BooleanField(initial=True, required=False)
    processing = forms.BooleanField(initial=False, required=False)
    failed = forms.BooleanField(initial=False, required=False)
    
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
    if req.GET:
        form_data = req.GET.copy()
        if not (req.GET.get('calibrated')
                or req.GET.get('processing')
                or req.GET.get('failed')):
            form_data['calibrated'] = 'on'
    else:
        form_data = None

    form = ImageSearchForm(form_data)
    context = {}
    images = UserImage.objects.all_visible()
    page_number = req.GET.get('page',1)
    category = 'tag'
    calibrated = True
    processing = False
    failed = False

    if form.is_valid(): 
        calibrated = form.cleaned_data.get('calibrated')
        processing = form.cleaned_data.get('processing')
        failed = form.cleaned_data.get('failed')

        category = form.cleaned_data.get('search_category');
        if category == 'tag':
            tags = form.cleaned_data.get('tags','')
            if tags.strip():
                images = UserImage.objects.none()
                tag_objs = []
                tags = map(strip,tags.split(','))
                tags = list(set(tags)) # remove duplicate tags
                
                images = UserImage.objects.all_visible().filter(tags__text__in=tags).distinct()
                tag_objs = Tag.objects.filter(text__in=tags)
                context['tags'] = tag_objs

        elif category == 'user':
            username = form.cleaned_data.get('user','')

            if username.strip():
                user = User.objects.filter(profile__display_name=username)[:1]
                images = UserImage.objects.none()
                if len(user) > 0:
                    images = UserImage.objects.all_visible().filter(user=user)
                    context['display_user'] = user[0] 
                else:
                    context['display_users'] = User.objects.filter(profile__display_name__startswith=username)[:5]
        elif category == 'location':
            ra = form.cleaned_data.get('ra', 0)
            dec = form.cleaned_data.get('dec', 0)
            radius = form.cleaned_data.get('radius', 0)

            if ra and dec and radius: 
                ra *= math.pi/180
                dec *= math.pi/180
                tempr = math.cos(dec)
                x = tempr*math.cos(ra)
                y = tempr*math.sin(ra)
                z = math.sin(dec)
                r = radius/180*math.pi
               
                # HACK - there's probably a better way to do this..?
                where = ('(x-(%(x)f))*(x-(%(x)f))+(y-(%(y)f))*(y-(%(y)f))+(z-(%(z)f))*(z-(%(z)f)) < (%(r)f)*(%(r)f)'
                        % dict(x=x,y=y,z=z,r=r))
                where2= '(r <= %f)' % r
                cals = Calibration.objects.extra(where=[where,where2])
                images = UserImage.objects.filter(jobs__calibration__in=cals)
        elif category == 'image':
            image_id = form.cleaned_data.get('image')
            if image_id:
                image = get_object_or_404(UserImage, pk=image_id)
                context['image'] = image
                images = image.get_neighbouring_user_images()

    if calibrated is False:
        images = images.exclude(jobs__status='S')
    if processing is False:
        images = images.exclude(jobs__status__isnull=True)
    if failed is False:
        images = images.exclude(jobs__status='F')

    page = get_page(images, 4*5, page_number)
    context.update({'form': form,
                    'search_category': category,
                    'image_page': page})
    return render_to_response('user_image/search.html',
        context,
        context_instance = RequestContext(req))
