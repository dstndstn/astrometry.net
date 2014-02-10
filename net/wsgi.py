import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'astrometry.net.settings')

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)
path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

path = os.path.dirname(__file__)
path = os.path.dirname(path)

os.environ['PATH'] += ':/usr/local/netpbm/bin:%s/blind:%s/util' % (path,path)

import logging
logfn = os.path.join(path, 'nova.log')
print >> sys.stderr, 'Logging to', logfn
logging.basicConfig(filename=logfn, level=logging.DEBUG)

from django.core.wsgi import get_wsgi_application
print >> sys.stderr, 'running django handler...'
application = get_wsgi_application()
print >> sys.stderr, 'application:', application
print >> sys.stderr, 'ran django handler.'
