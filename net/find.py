import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
import settings
from astrometry.net.models import *
from astrometry.util.file import *
from astrometry.util.multiproc import *
from log import *

import django
django.setup()

from django.contrib.auth.models import User

import logging
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)

def bounce_try_dojob(jobid):
    print 'Trying Job ID', jobid
    job = Job.objects.filter(id=jobid)[0]
    print 'Found Job', job
    return try_dojob(job, job.user_image)

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%(prog)')
    parser.add_option('-s', '--sub', type=int, dest='sub', help='Submission ID')
    parser.add_option('-j', '--job', type=int, dest='job', help='Job ID')
    parser.add_option('-u', '--userimage', type=int, dest='uimage', help='UserImage ID')
    parser.add_option('-r', '--rerun', dest='rerun', action='store_true',
                      help='Re-run this submission/job?')

    parser.add_option('--threads', type=int, help='Re-run failed jobs within this process using N threads; else submit to process_submissions process.')

    parser.add_option('--chown', dest='chown', type=int, default=0, help='Change owner of userimage by user id #')
    parser.add_option('--ssh', action='store_true', default=False,
              help='Find submissions whose jobs have ssh errors')
    parser.add_option('--minsub', type='int', default=0,
              help='Minimum submission id to look at')

    parser.add_option('--empty', action='store_true', default=False,
              help='Find submissions whose jobs have no log files')

    parser.add_option('--delete', action='store_true', default=False,
              help='Delete everything associated with the given image')

    opt,args = parser.parse_args()
    if not (opt.sub or opt.job or opt.uimage or opt.ssh or opt.empty):
        print 'Must specify one of --sub, --job, or --userimage (or --ssh or --empty)'
        parser.print_help()
        sys.exit(-1)

    if opt.ssh or opt.empty:
        subs = Submission.objects.all()
        if opt.minsub:
            subs = subs.filter(id__gt=opt.minsub)
        subs = subs.order_by('-id')
        failedsubs = []
        failedjobs = []
        for sub in subs:
            print 'Checking submission', sub.id
            allfailed = True
            # last failed Job
            failedjob = None
            uis = sub.user_images.all()
            for ui in uis:
                jobs = ui.jobs.all()
                for job in jobs:
                    print '  job', job.id
                    if job.status == 'S':
                        print '    -> succeeded'
                        allfailed = False
                        break
                    print '    error msg', job.error_message
                    logfn = job.get_log_file()
                    if not os.path.exists(logfn):
                        failedjob = job
                        continue

                    if opt.ssh:
                        log = read_file(logfn)
                        # 'Connection refused'
                        # 'Connection timed out'
                        if not 'ssh: connect to host astro.cs.toronto.edu port 22:' in log:
                            allfailed = False
                            break
                        print 'SSH failed'
                        failedjob = job

                    if opt.empty:
                        # log file found
                        allfailed = False
                        break

            if not allfailed:
                continue
            print 'All jobs failed for sub', sub.id #, 'via ssh failure'
            failedsubs.append(sub)
            failedjobs.append(failedjob)

        print 'Found total of', len(failedsubs), 'failed Submissions'
        if opt.rerun:
            from process_submissions import try_dosub, try_dojob
            if opt.threads is not None:
                mp = multiproc(opt.threads)
                args = []
                for j in failedjobs:
                    if j is None:
                        continue
                    args.append(j.id) #, j.user_image))
                mp.map(bounce_try_dojob, args)

            else:
                for sub in failedsubs:
                    print 'Re-trying sub', sub.id
                    try_dosub(sub, 1)
            

    if opt.sub:
        sub = Submission.objects.all().get(id=opt.sub)
        print 'Submission', sub
        if sub.disk_file is None:
            print '  no disk file'
        else:
            print 'Path', sub.disk_file.get_path()
        uis = sub.user_images.all()
        print 'UserImages:', len(uis)
        for ui in uis:
            print '  ', ui
            print '  with Jobs:', len(ui.jobs.all())
            for j in ui.jobs.all():
                print '    ', j

        if opt.rerun:
            from process_submissions import try_dosub
            print 'Re-trying sub', sub.id
            try_dosub(sub, 1)

        if opt.delete:
            print 'Deleting submission', sub
            sub.delete()

    if opt.job:
        job = Job.objects.all().get(id=opt.job)
        print 'Job', job
        print job.get_dir()
        print 'Status:', job.status
        print 'Error message:', job.error_message
        ui = job.user_image
	print 'Log file exists:', os.path.exists(job.get_log_file())
        print 'UserImage:', ui
        print 'User', ui.user
        im = ui.image
        print 'Image', im
        sub = ui.submission
        print 'Submission', sub
        print sub.disk_file.get_path()

        if opt.rerun:
            from process_submissions import try_dojob
            print 'Re-trying job', job.id
            try_dojob(job, ui)

    if opt.uimage:
        ui = UserImage.objects.all().get(id=opt.uimage)
        print 'UserImage', ui

        if opt.chown:
            user = User.objects.all().get(id=opt.chown)
            print 'User:', user
            print 'chowning', ui, 'to', user
            ui.user = user
            ui.save()
