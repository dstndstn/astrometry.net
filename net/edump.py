import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
from astrometry.net.models import *

v = 'v4'

ever = EnhanceVersion.objects.all().filter(name=v)
print 'vers={}'
for e in ever:
    print 'vers[%i] = EnhanceVersion.objects.get(id=%i)' % (e.id, e.id)

# for e in ever:
#     print 'e=%s; e.save()' % repr(e)
# 

ver = EnhanceVersion.objects.get(name=v)

eim = EnhancedImage.objects.all().filter(version=ver).select_related('wcs', 'cals')
print '# Get ', len(eim), 'EnhancedImage objects'
for im in eim:
    wcs = im.wcs
    if wcs is None:
        continue
    print 'w=%s; w.save();' % (repr(wcs))
    print (('eim=EnhancedImage(version=vers[%i], nside=%i, healpix=%i, wcs=w, id=%i);'
            + ' eim.save()') %
           (im.version.id, im.nside, im.healpix, im.id))

