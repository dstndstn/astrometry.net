from __future__ import absolute_import
sitename = 'nova'

# settings_nova.py
from astrometry.net.settings_common import *

# MULTI_HOSTS = ['nova3.astrometry.net',
#                'nova4.astrometry.net',
#                'nova5.astrometry.net',
#                'nova6.astrometry.net',]
MULTI_HOSTS = []

# from settings_social import *
# ENABLE_SOCIAL = True
# SOUTH_MIGRATION_MODULES.update(SOCIAL_MIGRATION)
# TEMPLATE_CONTEXT_PROCESSORS += SOCIAL_TEMPLATE_CONTEXT_PROCESSORS
# INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
# AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS

from astrometry.net.settings_social2 import *
ENABLE_SOCIAL2 = True
INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS
TEMPLATES[0]['OPTIONS']['context_processors'].extend(SOCIAL_TEMPLATE_CONTEXT_PROCESSORS)

USE_X_FORWARDED_HOST = True

#INSTALLED_APPS = list(INSTALLED_APPS) + ['social.apps.django_app.default']

sitename = 'nova'

TEMPDIR = '/data1/tmp'
DATABASES['default']['NAME'] = 'an-nova'

LOGGING['loggers']['django.request']['level'] = 'WARN'

import logging
logger = logging.getLogger('django.db.backends')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

SESSION_COOKIE_NAME = 'NovaAstrometrySession'

ssh_solver_config = 'an-nova'

try:
    SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
    SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
except:
    SOCIAL_AUTH_GITHUB_KEY    = None
    SOCIAL_AUTH_GITHUB_SECRET = None
    
