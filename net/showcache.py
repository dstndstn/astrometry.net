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

if __name__ == '__main__':
    cfs = CachedFile.objects.all()
    print 'Total of', cfs.count(), 'files cached'
    nbytes = 0
    for cf in cfs:
        df = cf.disk_file
        path = df.get_path()
        sz = file_size(path)
        print '  %-32s' % cf.key, '=>', path, '  (size: %i bytes)' % sz
        nbytes += sz
    print 'Total of', nbytes, 'bytes'
        
