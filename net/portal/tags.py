from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect, QueryDict
from django.core.urlresolvers import reverse
from django.template import Context, RequestContext, loader

from astrometry.net.portal.job import Job, Submission, Tag
#from astrometry.net.portal.views import get_status_url

@login_required
def job_add_tag(request):
    from astrometry.net.portal import views

    if not 'jobid' in request.POST:
        return HttpResponse('no jobid')
    jobid = request.POST['jobid']
    job = views.get_job(jobid)
    if not job:
        return HttpResponse('no such jobid')
    if not job.can_add_tag(request.user):
        return HttpResponse('not permitted')
    if not 'tag' in request.POST:
        return HttpResponse('no tag')
    txt = request.POST['tag']
    if not len(txt):
        return HttpResponse('empty tag')
    tag = Tag(job=job,
              user=request.user,
              machineTag=False,
              text=txt,
              addedtime=Job.timenow())
    if not tag.is_duplicate():
        tag.save()
    return HttpResponseRedirect(views.get_status_url(jobid))

@login_required
def job_remove_tag(request):
    from astrometry.net.portal import views

    if not 'tag' in request.GET:
        return HttpResponse('no tag')
    tagid = request.GET['tag']
    tag = Tag.objects.all().filter(id=tagid)
    if not len(tag):
        return HttpResponse('no such tag')
    tag = tag[0]
    if not tag.can_remove_tag(request.user):
        return HttpResponse('not permitted')
    tag.delete()
    return HttpResponseRedirect(views.get_status_url(tag.job.jobid))

@login_required
def taglist(request):
    from astrometry.net.portal import views

    mtags = Tag.objects.all().filter(machineTag=True).values('text').distinct()
    mtags = [d['text'] for d in mtags]

    utags = Tag.objects.all().filter(machineTag=False).values('text').distinct()
    utags = [d['text'] for d in utags]

    ctxt = {
        'usertags' : utags,
        'machinetags' : mtags,
        'view_tagtxt_url' : reverse(views.joblist) + '?type=tag&tagtext=',
        }
    t = loader.get_template('portal/taglist.html')
    c = RequestContext(request, ctxt)
    return HttpResponse(t.render(c))

