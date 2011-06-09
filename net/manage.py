#!/usr/bin/env python

# dstn -- add .. to PYTHONPATH
import sys
import os
path = os.path.abspath(__file__)
#print 'file:', __file__
#print 'path:', path
#print 'dirname:', os.path.dirname(path)
#print 'dirnamex2:', os.path.dirname(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
#print 'sys.path:', '\n  '.join(sys.path)

from django.core.management import execute_manager
import imp
try:
    imp.find_module('settings') # Assumed to be in the same directory.
except ImportError:
    import sys
    sys.stderr.write("Error: Can't find the file 'settings.py' in the directory containing %r. It appears you've customized things.\nYou'll have to run django-admin.py, passing it your settings module.\n" % __file__)
    sys.exit(1)

import settings

if __name__ == "__main__":
    execute_manager(settings)
