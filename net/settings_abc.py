from __future__ import absolute_import
sitename = 'abc'
from astrometry.net.settings_common import *

ALLOWED_HOSTS += ['abc.astrometry.net']

from astrometry.net.settings_social2 import *
ENABLE_SOCIAL2 = True
INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS
TEMPLATES[0]['OPTIONS']['context_processors'].extend(SOCIAL_TEMPLATE_CONTEXT_PROCESSORS)

USE_X_FORWARDED_HOST = True

sitename = 'abc'

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-abc'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'ABCAstrometrySession'

ssh_solver_config = 'an-abc'

try:
    SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
    SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
except:
    SOCIAL_AUTH_GITHUB_KEY    = None
    SOCIAL_AUTH_GITHUB_SECRET = None
    
