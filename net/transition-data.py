#! /usr/bin/env python
import os
import shutil
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
import settings
from astrometry.net.models import *

newdata = 'newdata'

subs = Submission.objects.all()
keepdfs = set()
print subs.count(), 'Submissions'
for sub in subs:
    print 'Submission', sub
    df = sub.disk_file
    print 'DiskFile', df
    keepdfs.add(df.id)

    uis = sub.user_images
    print uis.count(), 'UserImages'
    for ui in uis:
        print 'UserImage', ui
        im = ui.image
        print 'Image', im
        df = im.disk_file
        print 'DiskFile', df
        keepdfs.add(df.id)

keepdfs = list(keepdfs)
keepdfs.sort()
print len(keepdfs), 'DiskFiles to keep'
print DiskFile.objects.all().count(), 'total DiskFiles'
for df in keepdfs:
    oldpath = df.get_path()
    newdir = os.path.join(newdata, 'uploads', df.file_hash[:3])
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    newpath = os.path.join(newdir, df.file_hash)
    shutil.move(oldpath, newpath)
    
