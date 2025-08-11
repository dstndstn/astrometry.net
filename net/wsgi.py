from __future__ import print_function
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'astrometry.net.settings')

path = os.path.dirname(__file__)
if path not in sys.path:
    sys.path.append(path)
path = os.path.dirname(path)
if path not in sys.path:
    sys.path.append(path)

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
print('Path:', path)

os.environ['PATH'] += ':/usr/local/netpbm/bin:%s/solver:%s/util:%s/plot' % (path,path,path)

import logging
#logfn = os.path.join(path, 'net', 'nova.log')
logfn = '/data/nova/nova.log'
logfn = os.environ.get('WSGI_LOG_FILE', logfn)
if len(logfn):
    print('Logging to', logfn, file=sys.stderr)
    logging.basicConfig(filename=logfn, level=logging.INFO)
else:
    logging.basicConfig(level=logging.DEBUG)

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
            print('Calling app()', file=sys.stderr)
            print('environ:', environ, file=sys.stderr)
            print('start_response:', start_response, file=sys.stderr)
            print('my pid:', os.getpid(), file=sys.stderr)
            try:
                r = self.app(environ, start_response)
            except:
                print('app() raised exception', file=sys.stderr)
                import traceback
                print(traceback.format_exc(), file=sys.stderr)
                return None
            print(file=sys.stderr)
            print('app() returned:', r, file=sys.stderr)
            print(file=sys.stderr)
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
    import os
    try:
        #py3
        from io import StringIO
    except ImportError:
        #py2
        from cStringIO import StringIO


    def application(environ, start_response):
        headers = []
        headers.append(('Content-Type', 'text/plain'))
        write = start_response('200 OK', headers)
        input = environ['wsgi.input']
        output = StringIO()
        print("PID: %s" % os.getpid(), file=output)
        print("UID: %s" % os.getuid(), file=output)
        print("GID: %s" % os.getgid(), file=output)
        print(file=output)
        keys = environ.keys()
        keys.sort()
        for key in keys:
            print('%s: %s' % (key, repr(environ[key])), file=output)
        print(file=output)
        output.write(input.read(int(environ.get('CONTENT_LENGTH', '0'))))
    
        try:
            import astrometry.net.settings
            print('settings:', astrometry.net.settings, file=output)
            import astrometry.net.models
            print('models:', astrometry.net.models, file=output)
            pass
        except:
            import traceback
            print('Exception:', traceback.format_exc(), file=output)
    
        return [output.getvalue()]
