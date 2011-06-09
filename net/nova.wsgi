import os
import sys

print >> sys.stderr, 'nova.wsgi: sys.path is', '\n  '.join(sys.path)
print >> sys.stderr, 'nova.wsgi: PYTHONPATH is', os.environ['PYTHONPATH']

path = '/home/nova/nova'
if path not in sys.path:
    sys.path.append(path)
path = '/home/nova/nova/net'
if path not in sys.path:
    sys.path.append(path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import astrometry.net.settings

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()


