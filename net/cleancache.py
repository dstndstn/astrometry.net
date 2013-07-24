import os
import sys
if __name__ == '__main__':
    me = __file__
    path = os.path.realpath(me)
    sys.path.append(os.path.dirname(os.path.dirname(path)))
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
import settings
from astrometry.net.models import *
from astrometry.util.file import *


def clean_dfs():
    for df in DiskFile.objects.all().order_by('file_hash'):
        if os.path.exists(df.get_path()):
            continue
        print 'Does not exist:', df

        ocoll = df.collection
        for coll in ['cached', 'resized', 'uploaded', 'uploaded-gunzip', 'uploaded-untar']:
            df.collection = coll
            if os.path.exists(df.get_path()):
                print '--> found in', coll
                df.save()
                continue

        df.delete()
        # print '  image_set:', df.image_set.all()
        # for im in df.image_set.all():
        #     print '    uis:', im.userimage_set.all()
        # print '  submissions:', df.submissions.all()
        # print '  cached:', df.cachedfile_set.all()

if __name__ == '__main__':
    clean_dfs()

    cfs = CachedFile.objects.all()
    print 'Total of', cfs.count(), 'files cached'
    nbytes = 0
    for cf in cfs:
        df = cf.disk_file
        path = df.get_path()
        if not os.path.exists(path):
            print 'Path does not exist:', path
            print 'Other CachedFiles sharing this DiskFile:'
            df.cachedfile_set.all().delete()
            df.delete()
            #cf.delete()
            continue
        sz = file_size(path)
        print '  %-32s' % cf.key, '=>', path, '  (size: %i bytes)' % sz
        nbytes += sz
    print 'Total of', nbytes, 'bytes'
        
