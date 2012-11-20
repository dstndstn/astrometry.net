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

subs = Submission.objects.all()
keepdfs = set()
print subs.count(), 'Submissions'
for sub in subs:
    print 'Submission', sub
    df = sub.disk_file
    df.collection = Image.ORIG_COLLECTION
    df.save()
    print 'DiskFile', df
    keepdfs.add(df)
    uis = sub.user_images.all()
    print uis.count(), 'UserImages'
    for ui in uis:
        print 'UserImage', ui
        im = ui.image
        print 'Image', im
        df = im.disk_file
        df.collection = Image.ORIG_COLLECTION
        df.save()
        print 'DiskFile', df
        keepdfs.add(df)

        im2 = im.thumbnail
        if im2:
            print 'Thumbnail', im2
            df = im2.disk_file
            df.collection = Image.RESIZED_COLLECTION
            df.save()
            print 'DiskFile', df
            keepdfs.add(df)
        im2 = im.display_image
        if im2:
            print 'Display-size', im2
            df = im2.disk_file
            df.collection = Image.RESIZED_COLLECTION
            df.save()
            print 'DiskFile', df
            keepdfs.add(df)

dropdfs = set()
cached = CachedFile.objects.all()
print cached.count(), 'CachedFiles'
for c in cached:
    dropdfs.add(c.disk_file)

alldfs = dropdfs.union(keepdfs)
    
print len(alldfs), 'DiskFiles of', DiskFile.objects.all().count(), 'total accounted for'
print len(keepdfs), 'to keep'
print len(dropdfs), 'to drop'
            
keepdfs = list(keepdfs)
keepdfs.sort()
print len(keepdfs), 'DiskFiles to keep'
print DiskFile.objects.all().count(), 'total DiskFiles'
for df in keepdfs:
    oldpath = df.OLD_get_path()
    newpath = df.get_path()
    newdir = os.path.dirname(newpath)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    print 'Moving', oldpath, 'to', newpath
    if os.path.exists(newpath) and not os.path.exists(oldpath):
        print 'Already moved', oldpath, newpath
        continue
    try:
        shutil.move(oldpath, newpath)
    except Exception as e:
        print 'Failed to move', oldpath, 'to', newpath
        print e


jobs = Job.objects.all()
print jobs.count(), 'jobs'
for job in jobs:
    oldpath = job.OLD_get_dir()
    newpath = job.get_dir()
    newdir = os.path.dirname(newpath)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    if os.path.exists(newpath) and not os.path.exists(oldpath):
        print 'Already moved', oldpath, newpath
        continue
    print 'Moving', oldpath, 'to', newpath
    shutil.move(oldpath, newpath)
    

    
