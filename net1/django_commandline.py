# This file should be imported by any command-line tools that want
# to use Django capabilities.

import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net1.settings'
#sys.path.extend(['/home/gmaps/test',
#                 '/home/gmaps/django/lib/python2.4/site-packages'])

import astrometry.net1.settings as settings

os.environ['LD_LIBRARY_PATH'] = settings.UTIL_DIR
os.environ['PATH'] = ':'.join(['/bin', '/usr/bin', '/usr/local/bin',
                               settings.BLIND_DIR,
                               settings.UTIL_DIR,
                               ])
# This must match the Apache setting UPLOAD_DIR
os.environ['UPLOAD_DIR'] = settings.UPLOAD_DIR

