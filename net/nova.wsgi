import os
import sys

#print >> sys.stderr, 'nova.wsgi: sys.path is', '\n  '.join(sys.path)

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)
path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

#print >> sys.stderr, 'nova.wsgi: sys.path is', '\n  '.join(sys.path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import astrometry.net.settings

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()


if True:
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

