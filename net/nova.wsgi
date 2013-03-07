import os
import sys

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)
path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

path = os.path.dirname(__file__)
path = os.path.dirname(path)

os.environ['PATH'] += ':/usr/local/netpbm/bin:%s/blind:%s/util' % (path,path)

#import astrometry.net.settings
# DEBUG
#print >> sys.stderr, 'settings.ROOT_URLCONF', astrometry.net.settings.ROOT_URLCONF
#from django.conf import settings
#print >> sys.stderr, 'settings:', dir(settings)
#print >> sys.stderr, 'settings.ROOT_URLCONF', settings.ROOT_URLCONF

import logging
logfn = os.path.join(path, 'nova.log')
print >> sys.stderr, 'Logging to', logfn
logging.basicConfig(filename=logfn, level=logging.DEBUG)

try:
    import django.core.handlers.wsgi
    application = django.core.handlers.wsgi.WSGIHandler()
except:
    import traceback
    logging.error('Exception in WSGI handler:\n' + traceback.format_exc())

if False:
    # Copied from:http://code.google.com/p/modwsgi/wiki/DebuggingTechniques
    # Logging WSGI middleware
    import pprint
    class LoggingMiddleware:
        def __init__(self, application):
            self.__application = application

        def __call__(self, environ, start_response):
            errors = environ['wsgi.errors']
            pprint.pprint(('REQUEST', environ), stream=errors)

            def _start_response(status, headers):
                pprint.pprint(('RESPONSE', status, headers), stream=errors)
                return start_response(status, headers)
            
            return self.__application(environ, _start_response)

    application = LoggingMiddleware(application)

