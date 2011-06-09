
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseBadRequest, QueryDict
from django.shortcuts import render_to_response, get_object_or_404, redirect
from django.template import Context, RequestContext, loader
from django.contrib.auth.decorators import login_required

from astrometry.net.models import *
from astrometry.net import settings
from log import *
from django import forms
from django.http import HttpResponseRedirect
import shutil
import os, errno
import hashlib
import tempfile

from astrometry.util import image2pnm
from astrometry.util.run_command import run_command
      
def dashboard(request):
    return render_to_response("dashboard.html",
        {
        },
        context_instance = RequestContext(request))


@login_required
def get_api_key(request):
    try:
        prof = request.user.get_profile()
    except UserProfile.DoesNotExist:
        loginfo('Creating new UserProfile for', request.user)
        prof = UserProfile(user=request.user)
        prof.create_api_key()
        prof.save()
        
    return HttpResponse(str(prof))

class UploadFileForm(forms.Form):
    file  = forms.FileField()

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            sub = handle_uploaded_file(request, request.FILES['file'])
            return redirect(status, subid=sub.id)
    else:
        form = UploadFileForm()
    return render_to_response('upload.html', {'form': form, 'user': request.user },
        context_instance = RequestContext(request))

def status(req, subid=None):
    logmsg("Submissions: " + ', '.join([str(x) for x in Submission.objects.all()]))

    if subid is not None:
        # strip leading zeros
        subid = int(subid.lstrip('0'))
    sub = get_object_or_404(Submission, pk=subid)

    # Would be convenient to have an "is this a single-image submission?" function
    # (This is really "UserImage" status, not Submission status)

    #logmsg("UserImages: " + ', '.join(['%i'%s.id for s in sub.user_images.all()]))

    logmsg("UserImages:")
    for ui in sub.user_images.all():
        logmsg("  %i" % ui.id)
        for j in ui.jobs.all():
            logmsg("    job " + str(j))

    job = None
    calib = None
    jobs = sub.get_best_jobs()
    logmsg("Best jobs: " + str(jobs))
    if len(jobs) == 1:
        job = jobs[0]
        logmsg("Job: " + str(job) + ', ' + job.status)
        calib = job.calibration
        
    return render_to_response('status.html', { 'sub': sub, 'job': job, 'calib':calib, },
        context_instance = RequestContext(req))
    
def annotated_image(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    ui = job.user_image
    img = ui.image
    df = img.disk_file
    imgfn = df.get_path()
    wcsfn = os.path.join(job.get_dir(), 'wcs.fits')
    f,pnmfn = tempfile.mkstemp()
    os.close(f)
    (filetype, errstr) = image2pnm.image2pnm(imgfn, pnmfn)
    if errstr:
        logmsg('Error converting image file %s: %s' % (imgfn, errstr))
        return HttpResponse('plot failed')
    f,annfn = tempfile.mkstemp()
    os.close(f)
    cmd = 'plot-constellations -w %s -i %s -o %s -N -C -B' % (wcsfn, pnmfn, annfn)
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

def sdss_image(req, jobid=None):
    job = get_object_or_404(Job, pk=jobid)
    #ui = job.user_image
    #img = ui.image
    #df = img.disk_file
    #imgfn = df.get_path()
    wcsfn = os.path.join(job.get_dir(), 'wcs.fits')
    f,plotfn = tempfile.mkstemp()
    os.close(f)
    # http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?ra=179.6897098439353&dec=-0.4546214816666667&scale=0.79224&opt=&width=512&height=512

def handle_uploaded_file(req, f):
    logmsg('handle_uploaded_file: req=' + str(req))
    logmsg('handle_uploaded_file: req.session=' + str(req.session))
    #logmsg('handle_uploaded_file: req.session.user=' + str(req.session.user))
    logmsg('handle_uploaded_file: req.user=' + str(req.user))

    # get file onto disk
    file_hash = DiskFile.get_hash()
    temp_file_path = tempfile.mktemp()
    uploaded_file = open(temp_file_path, 'wb+')
    for chunk in f.chunks():
        uploaded_file.write(chunk)
        file_hash.update(chunk)
    uploaded_file.close()
    # move file into data directory
    DiskFile.make_dirs(file_hash.hexdigest())
    shutil.move(temp_file_path, DiskFile.get_file_path(file_hash.hexdigest()))
    # create and populate the database entry
    df = DiskFile(file_hash = file_hash.hexdigest(), size=0, file_type='')
    df.set_size_and_file_type()
    df.save()

    # HACK
    sub = Submission(user=req.user, disk_file=df, scale_type='ul', scale_units='degwidth')
    sub.original_filename = f.name
    sub.save()
    logmsg('Made Submission' + str(sub))

    return sub
