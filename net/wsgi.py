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

if True:
    from django.core.wsgi import get_wsgi_application
    #print >> sys.stderr, 'running django handler...'
    application = get_wsgi_application()
    #print >> sys.stderr, 'django app:', application

if False:
    class Wrapper(object):
        def __init__(self, app):
            self.app = app

        def __call__(self, environ, start_response):
            print >> sys.stderr, 'Calling app()'
            print >> sys.stderr, 'environ:', environ
            print >> sys.stderr, 'start_response:', start_response
            print >> sys.stderr, 'my pid:', os.getpid()
            try:
                r = self.app(environ, start_response)
            except:
                print >> sys.stderr, 'app() raised exception'
                import traceback
                print >> sys.stderr, traceback.format_exc()
                return None
            print >> sys.stderr
            print >> sys.stderr, 'app() returned:', r
            print >> sys.stderr
            return r

    realapplication = Wrapper(application)
    application = realapplication
        


# # From http://blog.dscpl.com.au/2010/03/improved-wsgi-script-for-use-with.html
# import settings
# import django.core.management
# #django.core.management.setup_environ(settings)
# django.conf.settings.configure()
# utility = django.core.management.ManagementUtility()
# command = utility.fetch_command('runserver')
# command.validate()
# import django.conf
# import django.utils
# django.utils.translation.activate(django.conf.settings.LANGUAGE_CODE)
# import django.core.handlers.wsgi
# application = django.core.handlers.wsgi.WSGIHandler()

if False:
    import cStringIO
    import os
    
    def application(environ, start_response):
        headers = []
        headers.append(('Content-Type', 'text/plain'))
        write = start_response('200 OK', headers)
        input = environ['wsgi.input']
        output = cStringIO.StringIO()
        print >> output, "PID: %s" % os.getpid()
        print >> output, "UID: %s" % os.getuid()
        print >> output, "GID: %s" % os.getgid()
        print >> output
        keys = environ.keys()
        keys.sort()
        for key in keys:
            print >> output, '%s: %s' % (key, repr(environ[key]))
        print >> output
        output.write(input.read(int(environ.get('CONTENT_LENGTH', '0'))))
    
        try:
            import astrometry.net.settings
            print >> output, 'settings:', astrometry.net.settings
            import astrometry.net.models
            print >> output, 'models:', astrometry.net.models
            pass
        except:
            import traceback
            print >> output, 'Exception:', traceback.format_exc()
    
        return [output.getvalue()]
