from __future__ import absolute_import
from .settings_common import *

from .settings_social import *
ENABLE_SOCIAL = True
SOUTH_MIGRATION_MODULES.update(SOCIAL_MIGRATION)
TEMPLATE_CONTEXT_PROCESSORS += SOCIAL_TEMPLATE_CONTEXT_PROCESSORS
INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-staging'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'StagingAstrometrySession'

ssh_solver_config = 'an-nova'
sitename = 'staging'

SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
