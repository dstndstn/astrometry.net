from __future__ import absolute_import
sitename = 'ccnova'

from astrometry.net.settings_common import *

HENRY_DRAPER_CAT = '/data/nova/hd.fits'
TYCHO2_KD = '/data/nova/tycho2.kd'
HIPPARCOS_CAT = '/data/nova/hip.fits'

from astrometry.net.settings_social2 import *
ENABLE_SOCIAL2 = True
INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS
TEMPLATES[0]['OPTIONS']['context_processors'].extend(SOCIAL_TEMPLATE_CONTEXT_PROCESSORS)

sitename = 'ccnova'

TEMPDIR = '/data/tmp'
DATABASES['default']['NAME'] = 'an-nova'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'CCNovaAstrometrySession'

#ssh_solver_config = 'an-nova2'

try:
    SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
    SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
except:
    SOCIAL_AUTH_GITHUB_KEY    = None
    SOCIAL_AUTH_GITHUB_SECRET = None
    
