#! /usr/bin/env python
import sys
import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('Adding', p)
sys.path.append(p)
import shutil
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
import settings
from astrometry.net.models import *

#readonly = True
readonly = False

readonlydb = False

subs = Submission.objects.select_related().all()
keepdfs = set()
print(subs.count(), 'Submissions')
for i,sub in enumerate(subs):
    if i % 1000 == 0:
        print('Submission', i, 'of', len(subs), ':', sub)
    #print 'Submission', i, 'of', len(subs), ':', sub
    df = sub.disk_file
    df.collection = Image.ORIG_COLLECTION
    if not readonlydb:
        df.save()
    #print 'DiskFile', df
    keepdfs.add(df)
    uis = sub.user_images.all()
    #print uis.count(), 'UserImages'
    for ui in uis:
        #print 'UserImage', ui
        im = ui.image
        #print 'Image', im
        df = im.disk_file
        df.collection = Image.ORIG_COLLECTION
        if not readonlydb:
            df.save()
        #print 'DiskFile', df
        keepdfs.add(df)

        im2 = im.thumbnail
        if im2:
            #print 'Thumbnail', im2
            df = im2.disk_file
            df.collection = Image.RESIZED_COLLECTION
            if not readonlydb:
                df.save()
            #print 'DiskFile', df
            keepdfs.add(df)
        im2 = im.display_image
        if im2:
            #print 'Display-size', im2
            df = im2.disk_file
            df.collection = Image.RESIZED_COLLECTION
            if not readonlydb:
                df.save()
            #print 'DiskFile', df
            keepdfs.add(df)

dropdfs = set()
cached = CachedFile.objects.all()
print(cached.count(), 'CachedFiles')
for c in cached:
    dropdfs.add(c.disk_file)

bothdfs = dropdfs.union(keepdfs)
    
print(len(bothdfs), 'DiskFiles of', DiskFile.objects.all().count(), 'total accounted for')
print(len(keepdfs), 'to keep')
print(len(dropdfs), 'to drop')

alldfs = set([df for df in DiskFile.objects.all()])
orphans = alldfs - bothdfs
print(len(orphans), 'orphans:')
orphans = list(orphans)
for df in orphans:
    print('  ', df)
    print('    images', df.image_set.all())
    for im in df.image_set.all():
        print('      ', im)
        print('      .thumbnail', im.thumbnail)
        print('      thumbnail of', im.image_thumbnail_set.all())
        print('      .display_image', im.display_image)
        print('      display of', im.image_display_set.all())
        print('      userimages:', im.userimage_set.all())
        for ui in im.userimage_set.all():
            print('        ', ui)
            print('        .user', ui.user)
            print('        .sub', ui.submission)
    print('    subs', df.submissions.all())
    print('    cached', df.cachedfile_set.all())

            
keepdfs = list(keepdfs)
keepdfs.sort()
print(len(keepdfs), 'DiskFiles to keep')
print(DiskFile.objects.all().count(), 'total DiskFiles')
missing = []
for df in keepdfs:
    oldpath = df.OLD_get_path()
    newpath = df.get_path()
    newdir = os.path.dirname(newpath)
    if not os.path.exists(newdir):
        if not readonly:
            os.makedirs(newdir)
        else:
            print('fake makedirs', newdir)
    print('Moving', oldpath, 'to', newpath)
    if os.path.exists(newpath) and not os.path.exists(oldpath):
        print('Already moved', oldpath, newpath)
        continue
    if readonly:
        print('fake move', oldpath, '->', newpath)
        if not os.path.exists(oldpath):
            missing.append(oldpath)
        #assert(os.path.exists(oldpath))
    else:
        try:
            shutil.move(oldpath, newpath)
        except Exception as e:
            print('Failed to move', oldpath, 'to', newpath)
            print(e)

print(len(missing), 'missing:')
for x in missing:
    print(x)

jobs = Job.objects.select_related().all()
print(jobs.count(), 'jobs')
missing = []
for job in jobs:
    oldpath = job.OLD_get_dir()
    newpath = job.get_dir()
    newdir = os.path.dirname(newpath)
    if not os.path.exists(newdir):
        if readonly:
            print('fake makedirs', newdir)
        else:
            os.makedirs(newdir)
    if os.path.exists(newpath) and not os.path.exists(oldpath):
        print('Already moved', oldpath, newpath)
        continue
    print('Moving', oldpath, 'to', newpath)
    if not readonly:
        shutil.move(oldpath, newpath)
    else:
        #assert(os.path.exists(oldpath))
        if not os.path.exists(oldpath):
            missing.append(oldpath)

print(len(missing), 'missing:')
for x in missing:
    print(x)
