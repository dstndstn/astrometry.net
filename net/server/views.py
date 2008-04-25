import os
import os.path
import tempfile

from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.template import Context, RequestContext, loader

from astrometry.net.server.log import log
from astrometry.net.server.models import QueuedJob, Worker, JobQueue, Index
from astrometry.net.portal.job import Job, Calibration
from astrometry.net.portal.wcs import TanWCS
from astrometry.net.util.run_command import run_command

def summary(request):
    log('test.')
    jobs = QueuedJob.objects.all()
    for j in jobs:
        log('job', j, ': to-do:', j.pretty_unstarted_work(), ', in-progress:', j.pretty_inprogress_work())
    
    ctxt = {
        'jobqueues': JobQueue.objects.all(),
        'queuedjobs': jobs,
        'workers': Worker.objects.all().order_by('hostname', 'processid'),
        'indexes': Index.objects.all().order_by('indexid', 'healpix'),
        }
    t = loader.get_template('server/summary.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

def get_input(request):
    log('test.')
    qjob = QueuedJob.objects.get(job__jobid=request.GET['jobid'],
                                q__queuetype='solve')
    path = qjob.job.get_filename('job.axy')
    f = open(path, 'rb')
    res = HttpResponse()
    res['Content-Type'] = 'application/octet-stream'
    res.write(f.read())
    f.close()
    return res

def real_set_results(request):
    qjob = QueuedJob.objects.get(job__jobid=request.GET['jobid'],
                                 q__queuetype='solve')
    job = qjob.job
    tardata = request.POST['tar']
    log('tardata is %i bytes long' % len(tardata))
    tardata = tardata.decode('base64_codec')
    log('decoded tardata is %i bytes long' % len(tardata))

    # HACK - pipes?
    (f, tarfile) = tempfile.mkstemp('', 'tar-%s-' % job.jobid)
    os.close(f)
    f = open(tarfile, 'wb')
    f.write(tardata)
    f.close()
    log('wrote to tarfile', tarfile)

    outdir = Job.s_get_job_dir(job.jobid)
    cmd = 'cd %s; tar xvf %s' % (outdir, tarfile)
    log('command:', cmd)
    (rtn, out, err) = run_command(cmd)
    if rtn:
        return HttpResponse('tar failed: %i,' % rtn + out + ', ' + err)

    log('set_results: in dir %s, files' % outdir, out.split('\n'))

    log('wcs')
    # Add WCS to database.
    wcsfile = job.get_filename('wcs.fits')
    wcs = TanWCS(file=wcsfile)
    wcs.save()

    log('calib')
    # HACK - need to make blind write out raw TAN, tweaked TAN, and tweaked SIP.
    # HACK - compute ramin, ramax, decmin, decmax.
    calib = Calibration(raw_tan = wcs)
    calib.save()

    (yestweak, tweakorder) = job.get_tweak()
    if yestweak:
        import pytweak.tweak as tweak
        indexrd = job.get_filename('index.rd.fits')
        indexxy = job.get_filename('index.xy.fits')
        fieldrd = job.get_filename('field.rd.fits')
        fieldxy = job.get_filename('job.axy')
        tweakedtan = job.get_filename('tweaked-tan.fits')
        
        tweak(wcsfile, indexrd, fieldxy, tweakedtan,
              indexxy, fieldrd, 1, True)
        wcs = TanWCS(file=tweakedtan)
        wcs.save()
        calib.tweaked_tan = wcs
        calib.save()

        tweakedsip = job.get_filename('tweaked.fits')

        tweak(tweakedtan, indexrd, fieldxy, tweakedsip,
              indexxy, fieldrd, job.tweakorder, True)
        wcs = SipWCS(file=tweakedsip)
        wcs.save()
        calib.sip = wcs
        calib.save()

    log('job')
    job.set_status('Solved')
    job.calibration = calib
    job.add_machine_tags()
    job.save()

    return HttpResponse('ok')

def set_results(request):
    log('set_results()')
    try:
        return real_set_results(request)
    except Error, e:
        log('error', e)
        raise e
