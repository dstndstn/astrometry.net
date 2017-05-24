from __future__ import absolute_import
# settings_test.py
from .settings_common import *

ENABLE_SOCIAL = False

DATABASES['default']['ENGINE'] = 'django.db.backends.sqlite3'
DATABASES['default']['NAME'] = 'django.sqlite3'


