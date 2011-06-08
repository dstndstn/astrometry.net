import os
import sys

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


