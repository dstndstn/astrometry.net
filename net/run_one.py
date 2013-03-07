
from process_submissions import *

#try_dojob(job, uimage)
#try_dosub(sub, max_retrie)

import django

def _dojob(j):
    django.db.connection.close()
    #j = Job.objects.get(id=jid)
    print 'Running', j
    try_dojob(j, j.user_image)

if __name__ == '__main__':
    #job = Job.objects.get(id=33685)
    #print 'Job', job
    #try_dojob(job, job.user_image)

    #for s in Submission.objects.all().order_by('-id'):
    #    for im in s.user_images.all():
    #        for j in im.jobs.all():

    args = []
    for j in Job.objects.filter(id__gt=40000).order_by('-id'):
        print 'Checking job', j.id
        if j.status != 'F':
            continue
        log = j.get_log_tail()
        if log is None:
            continue
        if 'mkdir: cannot create directory `job-nova-' in log:
            print 'Re-running:', j
            #try_dojob(j, j.user_image)
            #args.append(j.id)
            args.append(j)
            continue

        wcsfn = j.get_wcs_file()
        if os.system('grep -l RA---SIN %s' % wcsfn) == 0:
            print 'Job', j.id, 'has SIN WCS'
            args.append(j)
            continue

    print 'Running Job ids:', args
    from astrometry.util.multiproc import multiproc
    mp = multiproc(4)
    django.db.connection.close()
    mp.map(_dojob, args)
    #for j in args:
    #    _dojob(j)



def x():
    oldsubs = Submission.objects.filter(processing_started__isnull=False,
                                        processing_finished__isnull=True)
    print oldsubs.count(), 'submissions started but not finished'
    for sub in oldsubs:
        print 'Resetting the processing status of', sub
