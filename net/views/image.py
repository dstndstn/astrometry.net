from __future__ import print_function
import os
import math
import stat
import time
from datetime import datetime, timedelta

try:
    # py3
    from urllib.parse import urlencode
    # from urllib.request import urlretrieve
except ImportError:
    # py2
    from urllib import urlencode
    # from urllib import urlencode urlretrieve

if __name__ == '__main__':
    os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
    import django
    django.setup()

from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required
from django import forms
from django.contrib import messages

from astrometry.net.models import *
from astrometry.net import settings
from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net.sdss_image import plot_sdss_image

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command
from astrometry.util.file import *

from astrometry.net.models import License

from astrometry.net.views.comment import *
from astrometry.net.views.license import *
from astrometry.net.util import get_page, get_session_form
from astrometry.net.util import NoBulletsRadioSelect
from astrometry.net.views.tag import TagForm, TagSearchForm
from astrometry.net.views.license import LicenseForm

from astrometry.net.views.enhance import *

import json

# repeat this import to override somebody else's import of the datetime module
from datetime import datetime, timedelta

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
            'publicly_visible': NoBulletsRadioSelect(),
        }

    def __init__(self, user, *args, **kwargs):
        super(UserImageForm, self).__init__(*args, **kwargs)
        self.fields['album'].choices = [('', 'none')]
        self.fields['album'].initial = ''
        user_image = kwargs.get('instance')
        if user.is_authenticated:
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

    print('Displaying user_image', user_image_id)
    print('Job', job)
    print('Image', uimage.image)
    print('Submission', uimage.submission)
    print('Submission user_images:', list(uimage.submission.user_images.all()))
    print('Submission disk file:', uimage.submission.disk_file.get_path())
    print('Image disk file:', uimage.image.disk_file.get_path())

    #license_form = get_session_form(req.session, PartialLicenseForm)
    comment_form = get_session_form(req.session, PartialCommentForm)
    tag_form = get_session_form(req.session, TagForm)

    images = {}
    dim = uimage.image.get_display_image(tempfiles=req.tempfiles)
    images['original_display'] = reverse('serve_image', kwargs={'id':dim.id})
    images['original'] = reverse('serve_image', kwargs={'id':uimage.image.id})
    image_type = 'original'
    if job:
        if job.calibration:
            images['annotated_display'] = reverse('annotated_image', kwargs={'jobid':job.id,'size':'display'})
            images['annotated'] = reverse('annotated_image', kwargs={'jobid':job.id,'size':'full'})
            images['grid_display'] = reverse('grid_image', kwargs={'jobid':job.id,'size':'display'})
            images['grid'] = reverse('grid_image', kwargs={'jobid':job.id,'size':'full'})
            images['sdss_display'] = reverse('sdss_image', kwargs={'calid':job.calibration.id,'size':'display'})
            images['sdss'] = reverse('sdss_image', kwargs={'calid':job.calibration.id,'size':'full'})
            images['galex_display'] = reverse('galex_image', kwargs={'calid':job.calibration.id,'size':'display'})
            images['galex'] = reverse('galex_image', kwargs={'calid':job.calibration.id,'size':'full'})
            images['unwise_display'] = reverse('unwise_image', kwargs={'calid':job.calibration.id,'size':'display'})
            images['unwise'] = reverse('unwise_image', kwargs={'calid':job.calibration.id,'size':'full'})
            images['legacysurvey_display'] = reverse('legacysurvey_image', kwargs={'calid':job.calibration.id,'size':'display'})
            images['legacysurvey'] = reverse('legacysurvey_image', kwargs={'calid':job.calibration.id,'size':'full'})
            images['redgreen_display'] = reverse('red_green_image', kwargs={'job_id':job.id,'size':'display'})
            images['redgreen'] = reverse('red_green_image', kwargs={'job_id':job.id,'size':'full'})
            #images['enhanced_display'] = reverse('enhanced_image', kwargs={'job_id':job.id,'size':'display'})
            #images['enhanced'] = reverse('enhanced_image', kwargs={'job_id':job.id,'size':'full'})
            image_type = 'annotated'
        images['extraction_display'] = reverse('extraction_image', kwargs={'job_id':job.id,'size':'display'})
        images['extraction'] = reverse('extraction_image', kwargs={'job_id':job.id,'size':'full'})

    image_type = req.GET.get('image', image_type)
    if image_type in images:
        display_url = images[image_type + '_display']
        fullsize_url = images[image_type]
    else:
        display_url=''
        fullsize_url=''

    flags = Flag.objects.all()
    if req.user.is_authenticated:
        selected_flags = [flagged_ui.flag for flagged_ui in
            FlaggedUserImage.objects.filter(
                user_image=uimage,
                user=req.user,
            )
        ]
    else:
        selected_flags = None

    if job and job.calibration:
        parity = (calib.get_parity() < 0)
        wcs = calib.raw_tan
        if calib.tweaked_tan is not None:
            wcs = calib.tweaked_tan
        imgurl   = req.build_absolute_uri(images['original'])
        thumburl = req.build_absolute_uri(images['original_display'])

        fits = uimage.image.disk_file.is_fits_image()
        y = wcs.imageh - wcs.crpix2
        orient = wcs.get_orientation()

        logmsg('Parity', parity, 'FITS', fits, 'Orientation', orient)

        if parity:
            orient = 360. - orient

        wwturl = 'http://www.worldwidetelescope.org/wwtweb/ShowImage.aspx?reverseparity=%s&scale=%.6f&name=%s&imageurl=%s&credits=Astrometry.net+User+(All+Rights+Reserved)&creditsUrl=&ra=%.6f&dec=%.6f&x=%.1f&y=%.1f&rotation=%.2f&thumb=%s' % (parity, wcs.get_pixscale(), uimage.original_file_name, imgurl, wcs.crval1, wcs.crval2, wcs.crpix1, y, orient, thumburl)
    else:
        wwturl = None

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
        'images': json.dumps(images),
        'display_url': display_url,
        'fullsize_url': fullsize_url,
        'image_type': image_type,
        'flags': flags,
        'selected_flags': selected_flags,
        'wwt_url': wwturl,
    }

    if uimage.is_public() or (req.user.is_authenticated and uimage.user == req.user):
        template = 'user_image/view.html'
    #elif SharedHideable.objects.filter(shared_with=req.user.id, hideable=image).count():
    #    template = 'user_image/view.html'
    else:
        messages.error(req, "Sorry, you don't have permission to view this content.")
        template = 'user_image/permission_denied.html'
    return render(req, template, context)

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

            pro = get_user_profile(req.user)
            license,created = License.objects.get_or_create(
                default_license=pro.default_license,
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

def serve_image(req, id=None, image=None):
    if image is None:
        image = get_object_or_404(Image, pk=id)
    res = HttpResponse(content_type=image.get_mime_type())

    date = datetime.now() + timedelta(days=7)
    res['Expires'] = time.asctime(date.timetuple())
    mtime = os.stat(image.disk_file.get_path())[stat.ST_MTIME]
    res['Last-Modified'] = time.asctime(time.gmtime(mtime))
    if 'filename' in req.GET:
        res['Content-Disposition'] = 'filename=%s' % req.GET['filename']
    image.render(res, tempfiles=req.tempfiles)
    return res

def serve_thumbnail_image(req, id=None):
    image = get_object_or_404(Image, pk=id)
    thumb = image.get_thumbnail()
    if thumb is None:
        return HttpResponse('missing image file')
    return serve_image(req, image=thumb)

def grid_image(req, jobid=None, size='full'):
    from astrometry.plot.plotstuff import (Plotstuff,
                                           PLOTSTUFF_FORMAT_JPG,
                                           PLOTSTUFF_FORMAT_PPM,
                                           plotstuff_set_size_wcs,
    )
    job = get_object_or_404(Job, pk=jobid)
    ui = job.user_image
    img = ui.image
    if size == 'display':
        dimg = img.get_display_image(tempfiles=req.tempfiles)
        scale = float(dimg.width)/img.width
        img = dimg
    else:
        scale = 1.0

    wcsfn = job.get_wcs_file()
    pnmfn = img.get_pnm_path(tempfiles=req.tempfiles)
    outfn = get_temp_file(tempfiles=req.tempfiles)

    plot = Plotstuff()
    plot.wcs_file = wcsfn
    plot.outformat = PLOTSTUFF_FORMAT_JPG
    plot.outfn = outfn
    plot.scale_wcs(scale)
    plotstuff_set_size_wcs(plot.pargs)

    # plot image
    pimg = plot.image
    pimg.set_file(str(pnmfn))
    pimg.format = PLOTSTUFF_FORMAT_PPM
    plot.plot('image')

    grid = plot.grid
    ra,dec,radius = job.calibration.get_center_radecradius()
    steps = np.array([ 0.02, 0.05, 0.1, 0.2, 0.5,
                       1., 2., 5., 10., 15.,  30., 60. ])
    istep = np.argmin(np.abs(np.log(radius) - np.log(steps)))
    grid.declabelstep = steps[istep]
    nd = plot.count_dec_labels()
    if nd < 2:
        istep = max(istep-1, 0)
        grid.declabelstep = steps[istep]
    grid.decstep = grid.declabelstep
    plot.alpha = 1.
    plot.plot('grid')

    plot.alpha = 0.7
    grid.declabelstep = 0
    grid.decstep /= 2.
    plot.plot('grid')
    grid.decstep = 0

    # RA
    cosdec = np.cos(np.deg2rad(dec))
    istep = np.argmin(np.abs(np.log(radius/cosdec) - np.log(steps)))
    grid.ralabelstep = steps[istep] #min(istep+1, len(steps)-1)]
    nra = plot.count_ra_labels()
    if nra < 2:
        istep = max(istep-1, 0)
        grid.ralabelstep = steps[istep]
    grid.rastep = grid.ralabelstep
    plot.alpha = 1.
    plot.plot('grid')

    plot.alpha = 0.7
    grid.ralabelstep = 0
    grid.rastep /= 2.
    plot.plot('grid')

    plot.write()
    res = HttpResponse(open(outfn, 'rb'))
    res['Content-Type'] = 'image/jpeg'
    return res

def annotated_image(req, jobid=None, size='full'):
    job = get_object_or_404(Job, pk=jobid)
    ui = job.user_image
    img = ui.image
    if size == 'display':
        dimg = img.get_display_image(tempfiles=req.tempfiles)
        scale = float(dimg.width)/img.width
        img = dimg
    else:
        scale = 1.0

    wcsfn = job.get_wcs_file()
    pnmfn = img.get_pnm_path(tempfiles=req.tempfiles)
    annfn = get_temp_file(tempfiles=req.tempfiles)

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
        #args.append('--uzccat %s' % uzcfn)
        args.append('--abellcat %s' % abellfn)
        if hdfn:
            args.append('--hdcat %s' % hdfn)


    if rad < 0.25 and tycho2fn:
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
    res = HttpResponse(open(annfn, 'rb'))
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
    plotfn = get_temp_file(tempfiles=req.tempfiles)

    logmsg('onthesky_image: cal', cal, 'wcs', wcsfn, 'plot', plotfn)

    #
    wcs = anutil.Tan(wcsfn, 0)
    zoom = int(zoom)

    if zoom == 0:
        zoom = wcs.radius() < 15.
        plot_aitoff_wcs_outline(wcs, plotfn, zoom=zoom)
    elif zoom == 1:
        zoom = wcs.radius() < 1.5
        plot_wcs_outline(wcs, plotfn, zoom=zoom)
    elif zoom == 2:
        zoom = wcs.radius() < 0.15
        plot_wcs_outline(wcs, plotfn, width=3.6, grid=1, zoom=zoom,
                         zoomwidth=0.36, hd=True, hd_labels=False,
                         tycho2=False)
        # hd=True is too cluttered at this level
    elif zoom == 3:
        plot_wcs_outline(wcs, plotfn, width=0.36, grid=0.1, zoom=False,
                         hd=True, hd_labels=True, tycho2=True)
    else:
        return HttpResponse('invalid zoom')
    res = HttpResponse(open(plotfn, 'rb'))
    res['Content-Type'] = 'image/png'
    return res


# def galex_image(req, calid=None, size='full'):
#     from astrometry.util import util as anutil
#     from astrometry.plot import plotstuff as ps
#     from astrometry.net.galex_jpegs import plot_into_wcs
#     cal = get_object_or_404(Calibration, pk=calid)
#     key = 'galex_size%s_cal%i' % (size, cal.id)
#     df = CachedFile.get(key)
# 
#     if df is None:
#         wcsfn = cal.get_wcs_file()
#         plotfn = get_temp_file(tempfiles=req.tempfiles)
#         if size == 'display':
#             image = cal.job.user_image
#             dimg = image.image.get_display_image(tempfiles=req.tempfiles)
#             scale = float(dimg.width)/image.image.width
#         else:
#             scale = 1.0
# 
#         # logmsg('WCS filename:', wcsfn)
#         # logmsg('Plot filename:', plotfn)
#         # logmsg('Basedir:', settings.GALEX_JPEG_DIR)
#         # logmsg('Scale:', scale)
# 
#         plot_into_wcs(wcsfn, plotfn, basedir=settings.GALEX_JPEG_DIR, scale=scale)
#         # cache
#         logmsg('Caching key "%s"' % key)
#         df = CachedFile.add(key, plotfn)
#     else:
#         logmsg('Cache hit for key "%s"' % key)
#     res = HttpResponse(open(df.get_path(), 'rb'))
#     res['Content-Type'] = 'image/png'
#     return res


def legacysurvey_viewer_image(req, cal, size, layer):
    wcsfn = cal.get_wcs_file()

    from astrometry.util.util import Tan
    wcs = Tan(wcsfn)
    image = cal.job.user_image.image

    if size == 'display':
        dw = image.get_display_width()
        scale = float(dw)/image.width
        print('Dispaly width:', dw, 'full width:', image.width, 'scale', scale)
        wcs = wcs.scale(scale)

    else:
        scale = 1.0

    args = dict(crval1='%.6f' % wcs.crval[0],
                crval2='%.6f' % wcs.crval[1],
                crpix1='%.2f' % wcs.crpix[0],
                crpix2='%.2f' % wcs.crpix[1],
                cd11='%.6g' % wcs.cd[0],
                cd12='%.6g' % wcs.cd[1],
                cd21='%.6g' % wcs.cd[2],
                cd22='%.6g' % wcs.cd[3],
                imagew='%i' % int(wcs.imagew),
                imageh='%i' % int(wcs.imageh))
    urlargs = urlencode(args)
    #flip = image.is_jpeglike()
    #if not flip:
    #    urlargs += '&flip'
    url = 'https://legacysurvey.org/viewer/cutout-wcs/?layer=' + layer + '&' + urlargs
    print('Redirecting to URL', url)
    return HttpResponseRedirect(url)

def sdss_image(req, calid=None, size='full'):
    cal = get_object_or_404(Calibration, pk=calid)
    return legacysurvey_viewer_image(req, cal, size, 'sdss')

def galex_image(req, calid=None, size='full'):
    cal = get_object_or_404(Calibration, pk=calid)
    return legacysurvey_viewer_image(req, cal, size, 'galex')

def unwise_image(req, calid=None, size='full'):
    cal = get_object_or_404(Calibration, pk=calid)
    return legacysurvey_viewer_image(req, cal, size, 'unwise-neo6')

def legacysurvey_image(req, calid=None, size='full'):
    cal = get_object_or_404(Calibration, pk=calid)
    return legacysurvey_viewer_image(req, cal, size, 'ls-dr9')

def red_green_image(req, job_id=None, size='full'):
    from astrometry.plot.plotstuff import (Plotstuff,
                                           PLOTSTUFF_FORMAT_PNG,
                                           PLOTSTUFF_FORMAT_PPM,
                                           #plotstuff_set_size_wcs,
    )

    job = get_object_or_404(Job, pk=job_id)
    ui = job.user_image
    sub = ui.submission
    img = ui.image
    if size == 'display':
        dimg = img.get_display_image(tempfiles=req.tempfiles)
        scale = float(dimg.width)/img.width
        img = dimg
    else:
        scale = 1.0

    axyfn = job.get_axy_file()
    wcsfn = job.get_wcs_file()
    rdlsfn = job.get_rdls_file()
    pnmfn = img.get_pnm_path(tempfiles=req.tempfiles)
    exfn = get_temp_file(tempfiles=req.tempfiles)

    try:
        plot = Plotstuff()
        plot.wcs_file = wcsfn
        plot.outformat = PLOTSTUFF_FORMAT_PNG
        plot.outfn = exfn
        plot.scale_wcs(scale)
        plot.set_size_from_wcs()
        #plotstuff_set_size_wcs(plot.pargs)

        # plot image
        pimg = plot.image
        pimg.set_file(str(pnmfn))
        pimg.format = PLOTSTUFF_FORMAT_PPM
        plot.color = 'white'
        plot.alpha = 1.
        if sub.use_sextractor:
            xy = plot.xy
            xy.xcol = 'X_IMAGE'
            xy.ycol = 'Y_IMAGE'
        plot.plot('image')

        # plot red
        xy = plot.xy
        if hasattr(img, 'sourcelist'):
            # set xy offsets for source lists
            fits = img.sourcelist.get_fits_table(tempfiles=req.tempfiles)
            #xy.xoff = int(fits.x.min())
            #xy.yoff = int(fits.y.min())
            xy.xoff = 0.
            xy.yoff = 0.

        xy.set_filename(str(axyfn))
        xy.scale = scale
        plot.color = 'red'
        xy.nobjs = 200
        plot.lw = 2.
        plot.markersize = 6
        plot.plot('xy')

        # plot green
        rd = plot.radec
        rd.set_filename(str(rdlsfn))
        plot.color = 'green'
        plot.markersize = 4
        plot.plot('radec')

        plot.write()
    except:
        return HttpResponse("plot failed")

    res = StreamingHttpResponse(open(exfn, 'rb'))
    res['Content-Type'] = 'image/png'
    return res

def extraction_image(req, job_id=None, size='full'):
    from astrometry.plot.plotstuff import (Plotstuff,
                                           PLOTSTUFF_FORMAT_PNG,
                                           PLOTSTUFF_FORMAT_PPM)

    job = get_object_or_404(Job, pk=job_id)
    ui = job.user_image
    sub = ui.submission
    img = ui.image
    if size == 'display':
        dimg = img.get_display_image(tempfiles=req.tempfiles)
        scale = float(dimg.width)/img.width
        img = dimg
    else:
        scale = 1.0

    axyfn = job.get_axy_file()
    pnmfn = img.get_pnm_path(tempfiles=req.tempfiles)
    exfn = get_temp_file(tempfiles=req.tempfiles)

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
            fits = img.sourcelist.get_fits_table(tempfiles=req.tempfiles)
            #xy.xoff = int(fits.x.min())
            #xy.yoff = int(fits.y.min())
            xy.xoff = xy.yoff = 1.

        if sub.use_sextractor:
            xy.xcol = 'X_IMAGE'
            xy.ycol = 'Y_IMAGE'

        xy.set_filename(str(axyfn))
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
        import traceback
        traceback.print_exc()
        return HttpResponse("plot failed")

    res = HttpResponse(open(exfn, 'rb'))
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

    stats = ['S', 'F', '']
    if calibrated is False:
        stats.remove('S')
    if processing is False:
        stats.remove('')
    if failed is False:
        stats.remove('F')
    if len(stats) < 3:
        images = images.filter(jobs__status__in=stats)
    #print 'index 1:', images.query
    # the public_only() view already sorts them
    #images = images.order_by('-submission__submitted_on')
    #print 'index 2:', images.query
    page_number = req.GET.get('page', 1)
    page = get_page(images, 27, page_number)
    context.update({
        'image_page': page,
        'show_images_form': form,
    })
    return render(req, template_name, context)

def index_tag(req):
    images = UserImage.objects.public_only(req.user)
    #print 'index_tag 1:', images.query
    form = TagSearchForm(req.GET)
    tag = None
    if form.is_valid():
        query = form.cleaned_data.get('query')
        exact = form.cleaned_data.get('exact')
        if query:
            if exact:
                try:
                    tags = Tag.objects.filter(text__iexact=query)
                    if tags.count() > 1:
                        # More than one match: do case-sensitive query
                        ctags = Tag.objects.filter(text=query)
                        # note, 'text' is the primary key, so >1 shouldn't be possible
                        if len(ctags) == 1:
                            tag = ctags[0]
                        else:
                            # Uh, multiple case-insensitive matches but no case-sens
                            # matches.  Arbitrarily choose first case-insens
                            tag = tags[0]
                    else:
                        tag = tags[0]
                    images = images.filter(tags=tag)
                except Tag.DoesNotExist:
                    images = UserImage.objects.none()
            else:
                images = images.filter(tags__text__icontains=query)

    images = images.distinct()
    #print 'index_tag 2:', images.query

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
    return render(req, 'user_image/index_by_user.html', context)

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

    return render(req, 'user_image/image_set.html', context)

def wcs_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_wcs_file(), 'rb')
    res = HttpResponse(f)
    res['Content-Type'] = 'application/fits'
    res['Content-Disposition'] = 'attachment; filename=wcs.fits'
    return res

def rdls_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_rdls_file(), 'rb')
    res = HttpResponse(f)
    res['Content-Type'] = 'application/fits'
    res['Content-Disposition'] = 'attachment; filename=rdls.fits'
    return res

def axy_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_axy_file(), 'rb')
    res = HttpResponse(f)
    res['Content-Type'] = 'application/fits'
    res['Content-Disposition'] = 'attachment; filename=axy.fits'
    return res

def image_rd_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)

    extra_args = ''
    ui = job.user_image
    sub = ui.submission
    if sub.use_sextractor:
        extra_args = ' -X X_IMAGE -Y Y_IMAGE'
    wcsfn = job.get_wcs_file()
    axyfn = job.get_axy_file()
    rdfn = get_temp_file(tempfiles=req.tempfiles)
    cmd = 'wcs-xy2rd -w %s -i %s -o %s' % (wcsfn, axyfn, rdfn) + extra_args
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('wcs-xy2rd failed: out ' + out + ', err ' + err)
    from astrometry.util.fits import fits_table
    xy = fits_table(axyfn)
    rd = fits_table(rdfn)
    for c in xy.get_columns():
        rd.set(c, xy.get(c))
    rd.writeto(rdfn)

    res = HttpResponse(open(rdfn, 'rb'))
    res['Content-Type'] = 'application/fits'
    res['Content-Length'] = file_size(rdfn)
    res['Content-Disposition'] = 'attachment; filename=image-radec.fits'
    return res

def corr_file(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    f = open(job.get_corr_file(), 'rb')
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
        pnmfn = get_temp_file(tempfiles=req.tempfiles)
        fitsinfn = get_temp_file(tempfiles=req.tempfiles)
        cmd = 'image2pnm.py -i %s -o %s && an-pnmtofits %s > %s' % (infn, pnmfn, pnmfn, fitsinfn)
        logmsg('Running: ' + cmd)
        (rtn, out, err) = run_command(cmd)
        if rtn:
            logmsg('out: ' + out)
            logmsg('err: ' + err)
            return HttpResponse('image2pnm.py failed: out ' + out + ', err ' + err)
    outfn = get_temp_file(tempfiles=req.tempfiles)
    cmd = 'new-wcs -i %s -w %s -o %s -d' % (fitsinfn, wcsfn, outfn)
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('plot failed: out ' + out + ', err ' + err)
    logmsg('new-wcs completed, output file has length %i' % file_size(outfn))
    #res = HttpResponse(open(outfn, 'rb'))
    #res = HttpResponse(open(outfn, 'rb').read())
    res = StreamingHttpResponse(open(outfn, 'rb'))
    res['Content-Type'] = 'application/fits'
    res['Content-Length'] = file_size(outfn)
    res['Content-Disposition'] = 'attachment; filename=new-image.fits'
    return res

def kml_file(req, jobid=None):
    return HttpResponse('KMZ requests are off for now.  Post at https://groups.google.com/forum/#!forum/astrometry for help.')
    import tempfile
    import PIL.Image
    job = get_object_or_404(Job, pk=jobid)
    wcsfn = job.get_wcs_file()
    img = job.user_image.image
    df = img.disk_file

    # Convert SIP to TAN if necessary (wcs2kml can't handle SIP)
    import fitsio
    wcshdr = fitsio.read_header(wcsfn)
    #print('CTYPE1:', wcshdr.get('CTYPE1'))
    if wcshdr.get('CTYPE1') == 'RA---TAN-SIP':
        from astrometry.util.util import Sip, Tan
        wcs = Sip(wcshdr)
        #print('Sip:', wcs)
        wcs = Tan(wcs.wcstan)
        #print('Writing WCS', wcs)
        hdr = fitsio.FITSHDR()
        wcs.add_to_header(hdr)
        #print('Header:', hdr)
        hdr['EQUINOX'] = 2000.
        tmpwcs = get_temp_file(tempfiles=req.tempfiles)
        fitsio.write(tmpwcs, None, header=hdr, clobber=True)
        #print('Wrote temp TAN WCS header', tmpwcs)
        wcsfn = tmpwcs

    pnmfn = img.get_pnm_path(tempfiles=req.tempfiles)
    imgfn = get_temp_file(tempfiles=req.tempfiles)
    image = PIL.Image.open(pnmfn)
    image.save(imgfn, 'PNG')

    dirnm = tempfile.mkdtemp()
    req.tempdirs.append(dirnm)
    warpedimgfn = 'image.png'
    kmlfn = 'doc.kml'
    outfn = get_temp_file(tempfiles=req.tempfiles)
    cmd = ('cd %(dirnm)s'
           '; %(wcs2kml)s '
           '--input_image_origin_is_upper_left '
           '--fitsfile=%(wcsfn)s '
           '--imagefile=%(imgfn)s '
           '--kmlfile=%(kmlfn)s '
           '--outfile=%(warpedimgfn)s '
           '; zip -j - %(warpedimgfn)s %(kmlfn)s > %(outfn)s ' %
           dict(dirnm=dirnm, wcsfn=wcsfn, imgfn=imgfn, kmlfn=kmlfn,
                wcs2kml=settings.WCS2KML,
                warpedimgfn=warpedimgfn, outfn=outfn))
    logmsg('Running: ' + cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        logmsg('out: ' + out)
        logmsg('err: ' + err)
        return HttpResponse('kml generation failed: ' + err)

    res = HttpResponse(open(outfn, 'rb'))
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
    return redirect('user_image', user_image_id)

def hide(req, user_image_id):
    image = get_object_or_404(UserImage, pk=user_image_id)
    if req.user.is_authenticated and req.user == image.user:
        image.hide()
    return redirect('user_image', user_image_id)

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
                tags = [t.strip() for t in tags.split(',')]
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
                images = image.get_neighbouring_user_images(limit=None)

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
    return render(req, 'user_image/search.html', context)


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    # class Duck(object):
    #     pass
    # req = Duck()
    # onthesky_image(req, zoom=0, calid=1)

    if False:
        loc = SkyLocation()
        loc.nside = 16
        loc.healpix = 889
        import time
        t0 = time.time()
        locs = loc.get_neighbouring_user_images()
        t1 = time.time()
        locs = locs[:6]
        t2 = time.time()
        print(len(locs), 'locations found')
        t3 = time.time()
        print('get_neighbouring_user_image:', t1-t0)
        print('limit:', t2-t1)
        print('count:', t3-t2)
        import sys
        sys.exit(0)
    
    from django.test import Client
    c = Client()
    #r = c.get('/user_images/2676353')
    #r = c.get('/extraction_image_full/4005556')
    #r = c.get('/red_green_image_display/4515804')
    #r = c.get('/user_images/4470069/')
    # jobid
    #r = c.get('/annotated_display/6411716')
    #r = c.get('/thumbnail_of_image/12561093')
    #r = c.get('/user_images/5845514')
    #r = c.get('/sdss_image_display/4629768')
    #r = c.get('/user_images/1533706')
    #r = c.get('/kml_file/2646067?ignore=.kmz')
    #r = c.get('/new_fits_file/9797275')
    r = c.get('/image_rd_file/2646067')
    #print(r)
    with open('out.html', 'wb') as f:
        for x in r:
            f.write(x)
