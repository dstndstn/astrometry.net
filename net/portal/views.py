import datetime
import logging
import math
import os
import os.path
import random
import sha
import tempfile
import time

from urllib import urlencode

from django import newforms as forms

import django.contrib.auth as auth
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

from django.db import models
from django.http import HttpResponse, HttpResponseRedirect, QueryDict
from django.newforms import widgets, ValidationError, form_for_model
from django.template import Context, RequestContext, loader
from django.core.urlresolvers import reverse

import astrometry.net.portal.mercator as merc
from astrometry.net.portal.models import UserPreferences
from astrometry.net.portal.job import Job, Submission, DiskFile, Tag
from astrometry.net.portal.convert import convert, get_objs_in_field
from astrometry.net.portal.log import log
from astrometry.net.portal import tags
from astrometry.util.file import file_size
#from astrometry.net.vo.models import Image as voImage
from astrometry.net import settings
from astrometry.util import sip
from astrometry.util import healpix

def urlescape(s):
    return s.replace('&', '&amp;')

def get_status_url(jobid):
    return reverse(jobstatus, args=[jobid])

def get_file_url(job, fn, args=None):
    url = reverse(getfile, args=[job.jobid, fn])
    if args:
        url += '?' + args
    return url

def get_job(jobid):
    if jobid is None:
        return None
    jobs = Job.objects.all().filter(jobid=jobid)
    if len(jobs) != 1:
        #log('Found %i jobs, not 1' % len(jobs))
        return None
    job = jobs[0]
    return job

# a decorator for calls that need a valid jobid; puts the job in
# request.job
def needs_job(handler):
    def handle_request(request, *args, **kwargs):
        #log('Calling handler...')
        rtn = handler(request, *args, **kwargs)
        #log('Called handler.')
        return rtn
    return handle_request

def get_submission(subid):
    if subid is None:
        return None
    subs = Submission.objects.all().filter(subid=subid)
    if len(subs) != 1:
        return None
    return subs[0]

def getsessionjob(request):
    if not 'jobid' in request.session:
        log('no jobid in session')
        return None
    jobid = request.session['jobid']
    return get_job(jobid)

@login_required
def joblist(request):
    myargs = QueryDict('', mutable=True)

    sub = None
    subid = request.GET.get('subid')
    if subid:
        sub = get_submission(subid)
    job = None
    jobid = request.GET.get('jobid')
    if jobid:
        job = get_job(jobid)
    n = int(request.GET.get('n', '50'))
    start = int(request.GET.get('start', '0'))
    if n:
        end = start + n
    else:
        end = -1

    format = request.GET.get('format', 'html')

    kind = request.GET.get('type')
    if not kind in [ 'user', 'nearby', 'tag', 'sub' ]:
        kind = 'user'

    if sub:
        myargs['subid'] = sub.subid
    if job:
        myargs['jobid'] = job.jobid
    if start:
        myargs['start'] = start
    if n:
        myargs['n'] = n
    if format != 'html':
        myargs['format'] = format
    myargs['type'] = kind

    colnames = {
        'jobid' : 'Job Id',
        'status' : 'Status',
        'starttime' : 'Start Time',
        'finishtime' : 'Finish Time',
        'enqueuetime' : 'Enqueued Time',
        'radec' : '(RA, Dec)',
        'fieldsize' : 'Field size',
        'tags' : 'Tags',
        'desc' : 'Description',
        'objsin' : 'Objects',
        'thumbnail' : 'Thumbnail',
        'annthumb' : 'Annotated Thumbnail',
        'user' : 'User',
        # DEBUG
        'diskfile' : 'Diskfile',
        }

    allcols = colnames.keys()

    cols = ','.join(request.GET.getlist('cols'))
    if cols:
        cols = cols.split(',')
        okcols = []
        for c in cols:
            if c in allcols:
                okcols.append(c)
        cols = okcols

    ctxt = {}

    gmaps = ''

    ajaxupdate = False
    title = None
    # allow duplicate DiskFiles?
    duplicates = False

    if kind == 'user':
        uname = request.GET.get('user')
        if not uname:
            return HttpResponse('no user')
        users = User.objects.all().filter(username=uname)
        if users.count() == 0:
            return HttpResponse('no such user')
        user = users[0]
        myargs['user'] = user.username

        jobs = Job.objects.all().filter(submission__user=user,
                                        exposejob=True).order_by('-enqueuetime', '-starttime')
        
        N = jobs.count()
        if not cols:
            cols = [ 'thumbnail', 'jobid', 'status' ]
        title = 'Jobs submitted by <i>%s</i>' % user.username

    elif kind == 'nearby':
        if job is None:
            return HttpResponse('no job')
        tags = job.tags.all().filter(machineTag=True, text__startswith='hp:')
        if tags.count() != 1:
            return HttpResponse('no such tag')
        tag = tags[0]
        parts = tag.text.split(':')
        if len(parts) != 3:
            return HttpResponse('bad tag')
        nside = int(parts[1])
        hp = int(parts[2])
        #log('nside %i, hp %i' % (nside, hp))
        hps = get_neighbouring_healpixes(nside, hp)
        tagtxts = ['hp:%i:%i' % (nside,hp) for (nside,hp) in hps]
        tags = Tag.objects.all().filter(machineTag=True, text__in=tagtxts)
        N = tags.count()
        if not cols:
            cols = [ 'thumbnail', 'jobid', 'user' ]
            title = 'Jobs near <i>%s</i>' % job.jobid

    elif kind == 'tag':
        #if 'tag' in request.GET:
        #    tagid = request.GET['tag']
        #    tag = Tag.objects.all().filter(id=tagid)
        tagtxt = request.GET.get('tagtext')
        if not tagtxt:
            return HttpResponse('no tag')
        myargs['tagtext'] = tagtxt
        tags = Tag.objects.all().filter(text=tagtxt)
        if tags.count() == 0:
            return HttpResponse('no such tag')

        tags = tags.filter(job__duplicate=False)
        N = tags.count()

        if not cols:
            cols = [ 'thumbnail', 'jobid', 'user' ]
        title = 'Jobs tagged with <i>%s</i>' % tagtxt

    elif kind == 'sub':
        if sub is None:
            return HttpResponse('no sub')
        job = None
        jobs = sub.jobs.all().order_by('enqueuetime', 'starttime', 'jobid')
        N = jobs.count()
        gmaps = (reverse('astrometry.net.tile.views.index') +
                 '?submission=%s' % sub.get_id() +
                 '&layers=tycho,grid,userboundary&arcsinh')
        if not cols:
            cols = [ 'jobid', 'status', 'starttime', 'finishtime' ]
        ajaxupdate = True

    myargs['cols'] = ','.join(cols)

    if end > 0 and end < N:
        args = myargs.copy()
        args['start'] = end
        ctxt['nexturl'] = request.path + '?' + args.urlencode()
        args['start'] = start + n * ((N - start) / n)
        ctxt['lasturl'] = request.path + '?' + args.urlencode()
    if start > 0:
        args = myargs.copy()
        prev = max(0, start-n)
        args['start'] = prev
        ctxt['prevurl'] = request.path + '?' + args.urlencode()
        if prev != 0:
            args['start'] = 0
            ctxt['firsturl'] = request.path + '?' + args.urlencode()
        
    ctxt['firstnum'] = max(0, start) + 1
    ctxt['lastnum']  = end == -1 and N or min(N, end)
    ctxt['totalnum']  = N

    if kind == 'user':
        jobs = jobs[start:end]

    elif kind == 'nearby':
        tags = tags[start:end]
        jobs = [t.job for t in tags]

    elif kind == 'tag':
        tags = tags.order_by('addedtime')
        tags = tags[start:end]
        jobs = [t.job for t in tags]

    elif kind == 'sub':
        jobs = jobs[start:end]

    addcols = [c for c in allcols if c not in cols]

    rjobs = []
    for i, job in enumerate(jobs):
        rend = []
        jobn = start + i
        for c in cols:
            t = ''
            tdclass = 'c'
            if c == 'jobid':
                t = ('<a href="'
                     + urlescape(get_status_url(job.jobid))
                     + '">'
                     + job.jobid
                     + '</a>')
            elif c == 'starttime':
                t = job.format_starttime_brief()
            elif c == 'finishtime':
                t = job.format_finishtime_brief()
            elif c == 'enqueuetime':
                t = job.format_enqueuetime_brief()
            elif c == 'status':
                t = job.format_status()
            elif c == 'radec':
                if job.solved():
                    wcs = job.calibration.raw_tan
                    (ra,dec) = wcs.get_field_center()
                    t = '(%.2f, %.2f)' % (ra, dec)
            elif c == 'fieldsize':
                if job.solved():
                    wcs = job.calibration.raw_tan
                    (w, h, units) = wcs.get_field_size()
                    t = '%.2f x %.2f %s' % (w, h, units)
            elif c == 'tags':
                tags = job.tags.all().filter(machineTag=False).order_by('addedtime')
                t = ', '.join([tag.text for tag in tags])
            elif c == 'desc':
                t = job.description or ''
            elif c == 'objsin':
                if job.solved():
                    objs = get_objs_in_field(job, job.diskfile)
                    t = ', '.join([
                        ('<a href="' + request.path + '?'
                         + urlescape(urlencode({'type': 'tag',
                                                'tagtext': obj.encode('utf_8') }))
                         + '">' + obj + '</a>'
                         ) for obj in objs])
            elif c == 'thumbnail':
                t = ('<img src="'
                     + get_file_url(job, 'thumbnail')
                     + '" alt="Thumbnail" />')
            elif c == 'annthumb':
                if job.solved():
                    t = ('<img src="'
                         + get_file_url(job, 'annotation-thumb')
                         + '" alt="Thumbnail" />')
            elif c == 'user':
                t = ('<a href="'
                     + reverse(joblist) + urlescape('?type=user&user='
                     + job.get_user().username)
                     + '">'
                     + job.get_user().username
                     + '</a>')
            # DEBUG
            elif c == 'diskfile':
                t = job.diskfile.filehash

            rend.append((tdclass, c, t))
        rjobs.append((rend, job.jobid, jobn))

    if format == 'xml':
        res = HttpResponse()
        res['Content-type'] = 'text/xml'

        res.write('<submission subid="%s">\n' % subid)
        if kind == 'sub' and sub.is_finished():
            res.write('  <stop />\n')
        for (rend, jobid, jobn) in rjobs:
            res.write('  <job jobid="%s" n="%i">\n' % (jobid, jobn))
            for (tdclass, c, t) in rend:
                res.write('   <%s>%s</%s>\n' % (c, t, c))
            res.write('  </job>\n')
        res.write('</submission>\n')
        return res

    else:
        cnames = [colnames.get(c) for c in cols]

        columns = []
        for c,n in zip(cols, cnames):
            args = request.GET.copy()
            delcols = cols[:]
            delcols.remove(c)
            args['cols'] = ','.join(delcols)
            delurl = request.path + '?' + args.urlencode()
            columns.append((c, n, delurl))

        addcolumns = zip(addcols, [colnames[c] for c in addcols])

        ctxt.update({
            'thisurl' : request.path + '?' + myargs.urlencode(),
            'addcolumns' : addcolumns,
            'columns' : columns,
            'submission' : sub,
            'jobs' : jobs,
            'rjobs' : rjobs,
            'reload_time' : (len(jobs) < 2) and 2 or 15,
            'statusurl' : get_status_url(''),
            'gmaps' : gmaps,
            'title' : title,
            })
        if ajaxupdate:
            ctxt['xmlsummaryurl'] = request.get_full_path() + '&format=xml'

        t = loader.get_template('portal/joblist.html')
        c = RequestContext(request, ctxt)
        return HttpResponse(t.render(c))

@login_required
def submission_status_xml(request):
    subid = request.GET.get('subid')
    if subid is None:
        return HttpResponse('no subid')
    subs = Submission.objects.all().filter(subid=subid)
    if len(subs) != 1:
        return HttpResponse('%i subs' % len(subs))
    sub = subs[0]
    jobs = sub.jobs.all().order_by('starttime', 'jobid')
    res = HttpResponse()
    res['Content-type'] = 'text/xml'

    res.write('<submission subid="%s">\n' % subid)
    for job in jobs:
        res.write('  <job jobid="%s">\n' % job.jobid)
        res.write('    <jobid>%s</jobid>\n' % job.jobid)
        s = job.status
        if job.failurereason:
            s += ': ' + job.failurereason
        res.write('    <status>%s</status>\n' % s)
        res.write('    <start>%s</start>\n' % job.format_starttime_brief())
        res.write('    <finish>%s</finish>\n' % job.format_finishtime_brief())
        res.write('  </job>\n')
    res.write('</submission>\n')

    return res


@login_required
def submission_status(request, submission):
    jobs = submission.jobs.all().order_by('-starttime', '-enqueuetime', 'jobid')

    somesolved = False
    for job in jobs:
        log("job: " + str(job))
        if job.solved():
            somesolved = True
            log("somesolved = true.")
            break
        job.statusurl = get_status_url(job.jobid)

    #gmaps = (gmaps_config.gmaps_url + ('?submission=%s' % submission.jobid) +
    #         '&layers=tycho,grid,userboundary&arcsinh')

    gmaps = (reverse('astrometry.net.tile.views.index') +
             '?submission=%s' % submission.get_id() +
             '&layers=tycho,grid,userboundary&arcsinh')

    ctxt = {
        'submission' : submission,
        'reload_time' : (len(jobs) < 2) and 2 or 15,
        'jobs' : jobs,
        'somesolved' : somesolved,
        'gmaps_view_submission' : gmaps,
        'xmlsummaryurl' : reverse(submission_status_xml) + '?subid=' + submission.subid,
        }
    t = loader.get_template('portal/submission_status.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

def run_variant(request):
    return HttpResponse('Not implemented')

@login_required
def job_set_description(request):
    if not 'jobid' in request.POST:
        return HttpResponse('no jobid')
    jobid = request.POST['jobid']
    job = get_job(jobid)
    if not job:
        return HttpResponse('no such jobid')
    if job.get_user() != request.user:
        return HttpResponse('not your job')
    if not 'desc' in request.POST:
        return HttpResponse('no desc')
    desc = request.POST['desc']
    job.description = desc
    job.save()
    return HttpResponseRedirect(get_status_url(jobid))

def get_neighbouring_healpixes(nside, hp):
    hps = [ (nside, hp) ]
    # neighbours at this scale.
    neigh = healpix.get_neighbours(hp, nside)
    for n in neigh:
        hps.append((nside, n))
    # the next bigger scale.
    hps.append((nside-1, n/4))
    # the next smaller scale, plus neighbours.
    # (use a set because some of the neighbours are in common)
    allneigh = set()
    for i in range(4):
        n = healpix.get_neighbours(hp*4 + i, nside+1)
        allneigh.update(n)
    for i in allneigh:
        hps.append((nside+1, i))
    return hps

@needs_job
def jobstatus(request, jobid=None):
    log('jobstatus: jobid=', jobid)
    job = get_job(jobid)
    if not job:
        log('job not found.')
        submissions = Submission.objects.all().filter(subid=jobid)
        log('found %i submissions.' % len(submissions))
        if len(submissions):
            submission = submissions[0]
            log('submission: ', submission)
            jobs = submission.jobs.all()
            log('submission has %i jobs.' % len(jobs))
            if len(jobs) == 1:
                job = jobs[0]
                jobid = job.jobid
                log('submission has one job:', jobid)
                return HttpResponseRedirect(get_status_url(jobid))
            else:
                return submission_status(request, submission)
        else:
            return HttpResponse('no job with jobid ' + jobid)
    log('job found.')

    jobowner = (job.submission.user == request.user)
    anonymous = job.allowanonymous()
    if not (jobowner or anonymous):
        return HttpResponse('The owner of this job (' + job.submission.user.username + ') has not granted public access.')

    df = job.diskfile
    submission = job.submission
    log('jobstatus: Job is: ' + str(job))

    otherxylists = []
    # (image url, link url)
    #otherxylists.append(('test-image-url', 'test-link-url'))
    for n in (1,2,3,4):
        fn = convert(job, df, 'xyls-exists?', { 'variant': n })
        if fn is None:
            break
        otherxylists.append((get_file_url(job, 'sources-small', 'variant=%i' % n),
                             reverse(run_variant) + '?jobid=%s&variant=%i' % (job.jobid, n)))
                             #get_status_url(job.jobid) + '&run-xyls&variant=%i' % n))

    taglist = []
    for tag in job.tags.all().order_by('addedtime'):
        if tag.machineTag:
            continue
        tag.canremove = tag.can_remove_tag(request.user)
        taglist.append(tag)

    ctxt = {
        'jobid' : job.jobid,
        'jobstatus' : job.format_status_full(),
        'jobsolved' : job.solved(),
        'jobsubmittime' : submission.format_submittime(),
        'jobstarttime' : job.format_starttime(),
        'jobfinishtime' : job.format_finishtime(),
        'logurl' : get_file_url(job, 'blind.log'),
        'job' : job,
        'submission' : job.submission,
        'joburl' : (submission.datasrc == 'url') and submission.url or None,
        'jobfile' : (submission.datasrc == 'file') and submission.uploaded.userfilename or None,
        'jobscale' : job.friendly_scale(),
        'jobparity' : job.friendly_parity(),
        'needs_medium_scale' : job.diskfile.needs_medium_size(),
        'sources' : get_file_url(job, 'sources-medium'),
        'sources_big' : get_file_url(job, 'sources-big'),
        'sources_small' : get_file_url(job, 'sources-small'),
        'redgreen_medium' : get_file_url(job, 'redgreen'),
        'redgreen_big' : get_file_url(job, 'redgreen-big'),
        #'otherxylists' : otherxylists,
        'jobowner' : jobowner,
        'exposejob': job.is_exposed(),
        'tags' : taglist,
        'view_tagtxt_url' : reverse(joblist) + '?type=tag&tagtext=',
        'view_nearby_url' : reverse(joblist) + '?type=nearby&jobid=' + job.jobid,
        'view_user_url' : reverse(joblist) + '?type=user&user=',
        'set_description_url' : reverse(job_set_description),
        'add_tag_url' : reverse(tags.job_add_tag),
        'remove_tag_url' : reverse(tags.job_remove_tag) + '?',
        }

    if job.solved():
        wcsinfofn = convert(job, df, 'wcsinfo')
        f = open(wcsinfofn)
        wcsinfotxt = f.read()
        f.close()
        wcsinfo = {}
        for ln in wcsinfotxt.split('\n'):
            s = ln.split(' ')
            if len(s) == 2:
                wcsinfo[s[0]] = s[1]

        ctxt.update({'racenter' : '%.2f' % float(wcsinfo['ra_center']),
                     'deccenter': '%.2f' % float(wcsinfo['dec_center']),
                     'fieldw'   : '%.2f' % float(wcsinfo['fieldw']),
                     'fieldh'   : '%.2f' % float(wcsinfo['fieldh']),
                     'fieldunits': wcsinfo['fieldunits'],
                     'racenter_hms' : wcsinfo['ra_center_hms'],
                     'deccenter_dms' : wcsinfo['dec_center_dms'],
                     'orientation' : '%.3f' % float(wcsinfo['orientation']),
                     'pixscale' : '%.4g' % float(wcsinfo['pixscale']),
                     'parity' : (float(wcsinfo['det']) > 0 and 'Positive' or 'Negative'),
                     'wcsurl' : get_file_url(job, 'wcs.fits'),
                     'indexxyurl' : get_file_url(job, 'index.xy.fits'),
                     'indexrdurl' : get_file_url(job, 'index.rd.fits'),
                     'fieldxyurl' : get_file_url(job, 'field.xy.fits'),
                     'fieldrdurl' : get_file_url(job, 'field.rd.fits'),
                     })

        ctxt['objsinfield'] = get_objs_in_field(job, df)

        # deg
        fldsz = math.sqrt(df.imagew * df.imageh) * float(wcsinfo['pixscale']) / 3600.0

        url = (reverse('astrometry.net.tile.views.get_tile') +
               '?layers=tycho,grid,userboundary' +
               '&arcsinh&wcsfn=%s' % job.get_relative_filename('wcs.fits'))
        smallstyle = '&w=300&h=300&lw=3'
        largestyle = '&w=1024&h=1024&lw=5'
        steps = [ {              'gain':0,    'dashbox':0.1,   'center':False },
                  {'limit':18,   'gain':-0.5, 'dashbox':0.01,  'center':True, 'dm':0.05  },
                  {'limit':1.8,  'gain':0.25,                  'center':True, 'dm':0.005 },
                  ]
        zlist = []
        for last_s in range(len(steps)):
            s = steps[last_s]
            if 'limit' in s and fldsz > s['limit']:
                log('break')
                break
        else:
            last_s = len(steps)

        for ind in range(last_s):
            s = steps[ind]
            if s['center']:
                xmerc = float(wcsinfo['ra_center_merc'])
                ymerc = float(wcsinfo['dec_center_merc'])
                ralo = merc.merc2ra(xmerc + s['dm'])
                rahi = merc.merc2ra(xmerc - s['dm'])
                declo = merc.merc2dec(ymerc - s['dm'])
                dechi = merc.merc2dec(ymerc + s['dm'])
                bb = [ralo, declo, rahi, dechi]
            else:
                bb = [0, -85, 360, 85]
            urlargs = ('&gain=%g' % s['gain']) + ('&bb=' + ','.join(map(str, bb)))
            if (ind < (last_s-1)) and ('dashbox' in s):
                urlargs += '&dashbox=%g' % s['dashbox']

            zlist.append([url + smallstyle + urlargs,
                          url + largestyle + urlargs])

        # HACK
        fn = convert(job, df, 'fullsizepng')
        url = (reverse('astrometry.net.tile.views.index') +
               ('?zoom=%i&ra=%.3f&dec=%.3f&userimage=%s' %
                (int(wcsinfo['merczoom']), float(wcsinfo['ra_center']),
                 float(wcsinfo['dec_center']), job.get_relative_job_dir())))

        ctxt.update({
            'gmapslink' : url,
            'zoomimgs'  : zlist,
            'annotation': get_file_url(job, 'annotation'),
            'annotation_big' : get_file_url(job, 'annotation-big'),
            })

    else:
        logfn = job.get_filename('blind.log')
        if os.path.exists(logfn):
            f = open(logfn)
            logfiletxt = f.read()
            f.close()
            lines = logfiletxt.split('\n')
            lines = '\n'.join(lines[-16:])
            log('job not solved')

            ctxt.update({
                'logfile_tail' : lines,
                })

    t = loader.get_template('portal/status.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

def getdf(idstr):
    dfs = DiskFile.objects.all().filter(filehash=idstr)
    if not len(dfs):
        return None
    df = dfs[0]
    return df

def getfield(request):
    if not 'fieldid' in request.GET:
        return HttpResponse('no fieldid')
    fieldid = request.GET['fieldid']
    df = getdf(fieldid)
    if not df:
        return HttpResponse('no such field')
        
    if not df.show():
        return HttpResponse('The owner of this field has not granted public access.')
    #(' + field.user.username + ') 

    res = HttpResponse()
    ct = df.content_type()
    res['Content-Type'] = ct or 'application/octet-stream'
    fn = df.get_path()
    res['Content-Length'] = file_size(fn)
    #res['Content-Disposition'] = 'attachment; filename="' + f + '"'
    res['Content-Disposition'] = 'inline'
    f = open(fn)
    res.write(f.read())
    f.close()
    return res

def getfile(request, jobid=None, filename=None):
    job = get_job(jobid)
    if not job:
        return HttpResponse('no such job')

    jobowner = (job.get_user() == request.user)
    anonymous = job.is_exposed()
    if not (jobowner or anonymous):
        return HttpResponse('The owner of this job (' + job.get_user().username + ') has not granted public access.')

    res = HttpResponse()

    variant = 0
    if 'variant' in request.GET:
        variant = int(request.GET['variant'])

    pngimages = [ 'annotation', 'annotation-big', 'annotation-thumb',
                  'sources-small', 'sources-medium', 'sources-big',
                  'redgreen', 'redgreen-big', 'thumbnail',
                  ]

    convertargs = {}
    if variant:
        convertargs['variant'] = variant

    ### DEBUG - play with red-green colours.
    if filename.startswith('redgreen'):
        for x in ['red', 'green', 'rmarker', 'gmarker']:
            if x in request.GET:
                convertargs[x] = request.GET[x]

    if filename in pngimages:
        fn = convert(job, job.diskfile, filename, convertargs)
        res['Content-Type'] = 'image/png'
        res['Content-Length'] = file_size(fn)
        f = open(fn)
        res.write(f.read())
        f.close()
        return res

    binaryfiles = [ 'wcs.fits', 'match.fits', 'field.xy.fits', 'field.rd.fits',
                    'index.xy.fits', 'index.rd.fits' ]
    if filename in binaryfiles:
        downloadfn = filename
        if filename == 'field.xy.fits':
            fn = job.get_axy_filename()
        elif filename in [ 'index.xy.fits', 'field.rd.fits' ]:
            filename = convert(job, job.diskfile, filename, convertargs)
            fn = job.get_filename(filename)
        else:
            fn = job.get_filename(filename)
        res['Content-Type'] = 'application/octet-stream'
        res['Content-Disposition'] = 'attachment; filename="' + downloadfn + '"'
        res['Content-Length'] = file_size(fn)
        f = open(fn)
        res.write(f.read())
        f.close()
        return res

    textfiles = [ 'blind.log' ]
    if filename in textfiles:
        fn = job.get_filename(filename)
        res['Content-Type'] = 'text/plain'
        res['Content-Disposition'] = 'inline'
        res['Content-Length'] = file_size(fn)
        f = open(fn)
        res.write(f.read())
        f.close()
        return res

    if filename == 'origfile':
        if not job.is_exposed():
            return HttpResponse('access to this file is forbidden.')
        df = job.diskfile
        res = HttpResponse()
        ct = df.content_type()
        res['Content-Type'] = ct or 'application/octet-stream'
        fn = df.get_path()
        res['Content-Length'] = file_size(fn)
        # res['Content-Disposition'] = 'attachment; filename="' + f + '"'
        res['Content-Disposition'] = 'inline'
        f = open(fn)
        res.write(f.read())
        f.close()
        return res

    return HttpResponse('bad f')

def printvals(request):
    if request.POST:
        log('POST values:')
        for k,v in request.POST.items():
            log('  %s = %s' % (str(k), str(v)))
    if request.GET:
        log('GET values:')
        for k,v in request.GET.items():
            log('  %s = %s' % (str(k), str(v)))
    if request.FILES:
        log('FILES values:')
        for k,v in request.FILES.items():
            log('  %s = %s' % (str(k), str(v)))

@login_required
def userprefs(request):
    prefs = UserPreferences.for_user(request.user)
    PrefsForm = form_for_model(UserPreferences)
    if request.POST:
        form = PrefsForm(request.POST)
    else:
        form = PrefsForm({
            'exposejobs': prefs.exposejobs,
            })

    if request.POST and form.is_valid():
        prefs.set_expose_jobs(form.cleaned_data['exposejobs'])
        prefs.save()
        msg = 'Preferences Saved'
    else:
        msg = None
        
    ctxt = {
        'msg' : msg,
        'form' : form,
        }
    t = loader.get_template('portal/userprefs.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

@login_required
def summary(request):
    prefs = UserPreferences.for_user(request.user)
    submissions = Submission.objects.all().filter(user=request.user).order_by('-submittime')
    subs = []
    # keep track of multi-Job Submissions
    for sub in submissions:
        if len(sub.jobs.all()) > 1:
            subs.append(sub)

    jobs = []
    for sub in submissions:
        jobs += sub.jobs.all()

    #voimgs = voImage.objects.all().filter(user=request.user)
    voimgs = []

    log('Jobs:')
    for job in jobs:
        log('  %s: is_exposed: %s' % (job.get_id(), job.is_exposed()))

    for job in jobs:
        job.origfileurl = get_file_url(job, 'origfile')
        job.statusurl = get_status_url(job.jobid)

    ctxt = {
        'subs' : subs,
        'jobs' : jobs,
        'voimgs' : voimgs,
        'prefs' : prefs,

        'substatusurl' : reverse(joblist) + '?type=sub&subid=',
        }
    t = loader.get_template('portal/summary.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))
    
@login_required
def changeperms(request):
    if not request.POST:
        return HttpResponse('no POST')
    prefs = UserPreferences.for_user(request.user)

    log('changeperms:')
    for k,v in request.POST.items():
        log('  %s = %s' % (str(k), str(v)))
    expose = int(request.POST.get('exposejob', '-1'))
    if expose in [0, 1]:
        job = get_job(request.POST.get('jobid'))
        if job is None:
            return HttpResponse('no jobid')
        if job.get_user() != request.user:
            return HttpResponse('not your job!')
        log('exposejob = %i' % expose)
        job.set_exposed(expose)
        job.save()
    exposeall = int(request.POST.get('exposeall', '-1'))
    if exposeall in [0, 1]:
        #jobs = Job.objects.all().filter(submission.user=request.user)
        subs = Submission.objects.all().filter(user=request.user)
        for sub in subs:
            jobs = sub.jobs.all()
            for job in jobs:
                job.set_exposed(exposeall)
                job.save()

    if 'HTTP_REFERER' in request.META:
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    return HttpResponseRedirect(reverse(summary))

@login_required
def publishtovo(request):
    if not 'jobid' in request.POST:
        return HttpResponse('no jobid')
    job = get_job(request.POST['jobid'])
    if not job:
        return HttpResponse('no job')
    if job.submission.user != request.user:
        return HttpResponse('not your job')
    if not job.solved():
        return HttpResponse('job is not solved')
    wcs = job.tanwcs
    if not wcs:
        return HttpResponse('no wcs')


    # BIG HACK! - look through LD_LIBRARY_PATH if this is still needed...
    if not sip.libraryloaded():
        sip.loadlibrary('/home/gmaps/test/an-common/_sip.so')
    

    img = voImage(user = request.user,
                  field = job.field,
                  image_title = 'Field_%i' % (job.field.id),
                  instrument = '',
                  jdate = 0,
                  wcs = wcs,
                  )
    tanwcs = wcs.to_tanwcs()
    (ra, dec) = tanwcs.pixelxy2radec(wcs.imagew/2, wcs.imageh/2)
    (img.ra_center, img.dec_center) = (ra, dec)
    log('tanwcs: ' + str(tanwcs))
    log('(ra, dec) center: (%f, %f)' % (ra, dec))
    (ramin, ramax, decmin, decmax) = tanwcs.radec_bounds(10)
    (img.ra_min, img.ra_max, img.dec_min, img.dec_max) = (ramin, ramax, decmin, decmax)
    log('Saving vo.Image: ', img)
    img.save()

    return HttpResponse('Done.')
