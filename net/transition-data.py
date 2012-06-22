#! /usr/bin/env python
import sys
import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print 'Adding', p
sys.path.append(p)
import shutil
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
import settings
from astrometry.net.models import *

#NEW_DATADIR = 'newdata'
newdata = 'newdata'

subs = Submission.objects.all()
keepdfs = set()
print subs.count(), 'Submissions'
for sub in subs:
    print 'Submission', sub
    df = sub.disk_file
    print 'DiskFile', df
    keepdfs.add(df)
    uis = sub.user_images.all()
    print uis.count(), 'UserImages'
    for ui in uis:
        print 'UserImage', ui
        im = ui.image
        print 'Image', im
        df = im.disk_file
        print 'DiskFile', df
        keepdfs.add(df)

        im2 = im.thumbnail
        if im2:
            print 'Thumbnail', im2
            df = im2.disk_file
            print 'DiskFile', df
            keepdfs.add(df)
        im2 = im.display_image
        if im2:
            print 'Display-size', im2
            df = im2.disk_file
            print 'DiskFile', df
            keepdfs.add(df)

keepdfs = list(keepdfs)
keepdfs.sort()
print len(keepdfs), 'DiskFiles to keep'
print DiskFile.objects.all().count(), 'total DiskFiles'
for df in keepdfs:
    oldpath = df.get_path()
    #newdir = os.path.join(newdata, 'uploads', df.file_hash[:3])
    newpath = df.NEW_get_path()
    #newpath = newpath.replace(settings.DATADIR, os.path.join(newdata, 'data'))
    newdir = os.path.dirname(newpath)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    #newpath = os.path.join(newdir, df.file_hash)
    print 'Moving', oldpath, 'to', newpath
    if os.path.exists(newpath) and not os.path.exists(oldpath):
        print 'Already moved', oldpath, newpath
        continue
    shutil.move(oldpath, newpath)
    


jobs = Job.objects.all()
print jobs.count(), 'jobs'
for job in jobs:
    oldpath = job.get_dir()
    #jtxt = '%08i' % job.id
    #newdir = os.path.join(newdata, 'jobs', jtxt[:4])
    newpath = job.NEW_get_dir()
    newdir = os.path.dirname(newpath)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    #newpath = os.path.join(newdir, jtxt)
    if os.path.exists(newpath) and not os.path.exists(oldpath):
        print 'Already moved', oldpath, newpath
        continue
    print 'Moving', oldpath, 'to', newpath
    shutil.move(oldpath, newpath)
    

keepdfs = set(keepdfs)
cached = CachedFile.objects.all()
print cached.count(), 'CachedFiles'
for c in cached:
    keepdfs.add(c.disk_file)

print len(keepdfs), 'DiskFiles of', DiskFile.objects.all().count(), 'total accounted for'

    
