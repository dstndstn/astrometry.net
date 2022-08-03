from __future__ import print_function
from __future__ import absolute_import
import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
import settings

import django
django.setup()

from astrometry.net.models import *
from astrometry.util.file import *
from astrometry.util.multiproc import *
from log import *

from django.contrib.auth.models import User

import logging
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)

def delete_user(u):
    print('Deleting user', u)
    u.user_images.all().delete()
    u.submissions.all().delete()
    u.albums.all().delete()
    u.comments_left.all().delete()
    try:
        u.profile.delete()
    except:
        import traceback
        traceback.print_exc()
    u.delete()
    print('Deleted user', u)

def bounce_try_dojob(X):
    jobid, solve_command, solve_locally = X
    try:
        from process_submissions import try_dojob
        print('Trying Job ID', jobid)
        job = Job.objects.filter(id=jobid)[0]
        print('Found Job', job)
        r = try_dojob(job, job.user_image, solve_command, solve_locally)
        print('Job result for', job, ':', r)
        return r
    except:
        import traceback
        traceback.print_exc()

def main():
    import optparse
    parser = optparse.OptionParser('%(prog)')
    parser.add_option('-s', '--sub', type=int, dest='sub', help='Submission ID')
    parser.add_option('-j', '--job', type=int, dest='job', help='Job ID')
    parser.add_option('-u', '--userimage', type=int, dest='uimage', help='UserImage ID')
    parser.add_option('-e', '--email', help='Find user id with given email address')
    parser.add_option('-U', '--userid', help='Find user with given numerical ID', type=int)
    parser.add_option('-i', '--image', type=int, dest='image', help='Image ID')
    parser.add_option('-d', '--disk-file', type=str, dest='df', help='DiskFile id')
    parser.add_option('-r', '--rerun', dest='rerun', action='store_true',
                      help='Re-run this submission/job?')
    parser.add_option('--list-socials', default=False, action='store_true')
    parser.add_option('--list-profiles', default=False, action='store_true')
    
    parser.add_option('--threads', type=int, help='Re-run failed jobs within this process using N threads; else submit to process_submissions process.')

    parser.add_option('--chown', dest='chown', type=int, default=0, help='Change owner of userimage or submission by user id #')

    parser.add_option('--solve-command',
                      help='Command to run instead of ssh to actually solve image')
    parser.add_option('--solve-locally',
                      help='Command to run astrometry-engine on this machine, not via ssh')

    parser.add_option('--ssh', action='store_true', default=False,
              help='Find submissions whose jobs have ssh errors')
    parser.add_option('--minsub', type='int', default=0,
              help='Minimum submission id to look at')

    parser.add_option('--empty', action='store_true', default=False,
              help='Find submissions whose jobs have no log files')

    parser.add_option('--delete', action='store_true', default=False,
              help='Delete everything associated with the given job/submission/image')

    parser.add_option('--delete-sub', action='store_true', default=False,
                      help='When deleting a Job/UserImage, also delete the Submission?')

    parser.add_option('--deluser', action='store_true', default=False,
                      help='Delete everything associated with the given user')

    parser.add_option('--delextra', action='store_true', default=False,
                      help='Delete extraneous duplicate jobs?')
    
    parser.add_option('--hide', action='store_true', default=False,
                      help='For a UserImage, set publicly_visible=False')
    
    opt,args = parser.parse_args()

    if opt.list_socials:
        #from social.apps.django_app.default.models import UserSocialAuth
        ## ????
        from social_django.models import UserSocialAuth
        socs = UserSocialAuth.objects.order_by('created')
        for soc in socs:
            print('soc for user', soc.user.id, soc.user, 'provider', soc.provider, 'uid', soc.uid, 'extra', soc.extra_data)
        sys.exit(0)
    if opt.list_profiles:
        pros = UserProfile.objects.all()
        for pro in pros:
            print('userid', pro.user.id, 'display', pro.display_name, 'apikey', pro.apikey)
        sys.exit(0)

    if opt.email:
        users = User.objects.filter(email__contains=opt.email)
        print('Users with email containing "%s":' % opt.email)
        for u in users:
            print(u.id, u, u.email)
        if opt.deluser:
            delete_user(u)
        sys.exit(0)

    if opt.userid:
        users = User.objects.filter(id=opt.userid)
        print('Users with ID', opt.userid)
        for u in users:
            print(u.id, u, u.email)
            print(u.profile)
            print(u.profile)
            for k in ['email', 'first_name', 'last_name', 'profile', 'social_auth', 'username']:
                print(' ',k,getattr(u,k))
            for f in ['get_full_name', 'get_short_name', 'get_username',]:
                print(' ',f,getattr(u,f)())

            print(' submissions:')
            for s in u.submissions.all().order_by('-id'):
                print('    ', s.id)
        if opt.deluser:
            delete_user(u)
        sys.exit(0)
        
    if not (opt.sub or opt.job or opt.uimage or opt.image or opt.ssh or opt.empty or opt.df):
        print('Must specify one of --sub, --job, or --userimage or --image (or --ssh or --empty)')

        parser.print_help()
        sys.exit(-1)

    if opt.threads is not None:
        mp = multiproc(opt.threads)
    else:
        mp = None
        
    if opt.ssh or opt.empty or opt.delextra:
        subs = Submission.objects.all()
        if opt.minsub:
            subs = subs.filter(id__gt=opt.minsub)
        subs = subs.order_by('-id')
        failedsubs = []
        failedjobs = []
        for sub in subs:
            print('Checking submission', sub.id)
            allfailed = True
            # last failed Job
            failedjob = None
            uis = sub.user_images.all()
            for ui in uis:
                jobs = ui.jobs.all()
                for job in jobs:
                    print('  job', job.id)
                    if job.status == 'S':
                        print('    -> succeeded')
                        allfailed = False
                        break
                    print('    error msg', job.error_message)
                    logfn = job.get_log_file()
                    if not os.path.exists(logfn):
                        failedjob = job
                        continue

                    if opt.ssh:
                        log = read_file(logfn)
                        # 'Connection refused'
                        # 'Connection timed out'
                        if not (('ssh: connect to host astro.cs.toronto.edu port 22:' in log) or
                                ('ssh: Could not resolve' in log)):
                            allfailed = False
                            break
                        print('SSH failed')
                        failedjob = job

                    if opt.empty:
                        # exists but empty?
                        if file_size(logfn) == 0:
                            failedjob = job
                            continue
                        # log file found
                        allfailed = False
                        break

                if opt.delextra:
                    print('Delextra:', len(jobs), 'jobs', len(uis), 'uis; failedjob:', failedjob)
                    if len(jobs) > 1 and failedjob is not None:
                        print('Delextra: delete', failedjob)

            if not allfailed:
                continue
            print('All jobs failed for sub', sub.id) #, 'via ssh failure')
            #failedsubs.append(sub)
            failedjobs.append(failedjob)

        print('Found total of', len(failedsubs), 'failed Submissions and', len(failedjobs), 'failed Jobs')
        if opt.rerun:
            from process_submissions import try_dosub, try_dojob
            import logging
            logformat = '%(message)s'
            logging.basicConfig(level=logging.DEBUG, format=logformat)

            if opt.threads is not None:
                args = []
                for j in failedjobs:
                    if j is None:
                        continue
                    args.append((j.id, opt.solve_command, opt.solve_locally))
                mp.map(bounce_try_dojob, args)
            else:
                for job in failedjobs:
                    if job is None:
                        continue
                    print('Re-trying job', job.id)
                    try_dojob(job, job.user_image, opt.solve_command, opt.solve_locally)

            # FIXME -- failed subs??
            # 
            # else:
            #     for sub in failedsubs:
            #         print('Re-trying sub', sub.id)
            #         try_dosub(sub, 1)

    if opt.df:
        dfs = DiskFile.objects.filter(file_hash=opt.df)
        print('Found', dfs.count(), 'matching DiskFiles')
        for df in dfs:
            print('  ', df)
            print('    ', df.get_path())
            ims = Image.objects.filter(disk_file=df)
            print('  Used by', ims.count(), 'Image objects')
            for im in ims:
                print('    ', im, im.id)
                print('      size %i x %i' % (im.width, im.height))
                print('      thumb Image', im.thumbnail, (im.thumbnail and im.thumbnail.id or ''))
                print('      display Image', im.display_image, (im.display_image and im.display_image.id or ''))

            ims = Image.objects.filter(disk_file=df,
                                       display_image__isnull=False,
                                       thumbnail__isnull=False)
            print('  found', ims.count(), 'Images for this df w/ display and thumbnails')
                
    if opt.sub:
        sub = Submission.objects.all().get(id=opt.sub)
        print('Submission', sub)
        if sub.disk_file is None:
            print('  no disk file')
        else:
            print('Path', sub.disk_file.get_path())
            print('Is fits image:', sub.disk_file.is_fits_image())
            print('Is fits image:', sub.disk_file.file_type)
        print('User:', sub.user)
        uis = sub.user_images.all()
        print('UserImages:', len(uis))
        for ui in uis:
            print('  ', ui)
            print('  user', ui.user)
            print('  with Jobs:', len(ui.jobs.all()))
            for j in ui.jobs.all():
                print('    ', j)

        if opt.chown:
            newuser = User.objects.all().get(id=opt.chown)
            for ui in uis:
                ui.user = newuser
                ui.save()
            sub.user = newuser
            sub.save()

        if opt.rerun:
            from process_submissions import try_dosub, dosub
            print('Re-trying sub', sub.id)
            #try_dosub(sub, 1)
            dosub(sub)
            
        if opt.delete:
            print('Deleting submission', sub)
            sub.delete()

    if opt.job:
        job = Job.objects.all().get(id=opt.job)
        print('Job', job)
        print(job.get_dir())
        print('Status:', job.status)
        print('Error message:', job.error_message)
        ui = job.user_image
        print('UserImage:', ui)
        print('User', ui.user)
        im = ui.image
        print('Image', im)
        sub = ui.submission
        print('Submission', sub)
        print(sub.disk_file.get_path())

        if opt.rerun:
            from astrometry.net.process_submissions import try_dojob
            print('Re-trying job', job.id)
            try_dojob(job, ui, opt.solve_command, opt.solve_locally)

        if opt.delete:
            print('Deleting job', job)
            job.delete()

            if opt.delete_sub:
                print('Deleting submission', sub)
                sub.delete()

    if opt.uimage:
        ui = UserImage.objects.all().get(id=opt.uimage)
        print('UserImage', ui)
        im = ui.image
        print('Image', im)
        sub = ui.submission
        print('User', ui.user)
        print('Submission', sub)
        print(sub.disk_file.get_path())

        if opt.chown:
            user = User.objects.all().get(id=opt.chown)
            print('User:', user)
            print('chowning', ui, 'to', user)
            ui.user = user
            ui.save()

        if opt.delete:
            print('Deleting ui', ui)
            ui.delete()
        if opt.delete_sub:
            print('Deleting sub', sub)
            sub.delete()
        if opt.hide:
            print('Hiding ui', ui)
            ui.hide()

    if opt.image:
        im = Image.objects.all().get(id=opt.image)
        # thumbnail
        # display_image
        print('Image:', im, im.id)

        #uis = im.userimage_set()

        uis = UserImage.objects.all().filter(image=im.id)
        print('UserImages:', uis)

        print('Thumbnail:', im.thumbnail)
        print('Display:', im.display_image)
        
        if opt.delete:
            print('Deleting...')
            im.delete()
            if im.thumbnail:
                im.thumbnail.delete()
            if im.display_image:
                im.display_image.delete()
                
if __name__ == '__main__':
    main()
    
