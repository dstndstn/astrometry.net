import os
import sys

#print >> sys.stderr, 'nova.wsgi: sys.path is', '\n  '.join(sys.path)

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)
path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

print >> sys.stderr, 'nova.wsgi: sys.path is', '\n  '.join(sys.path)


os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import astrometry.net.settings

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()


