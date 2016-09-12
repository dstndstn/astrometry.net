from __future__ import absolute_import
from .settings_common import *

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-bbc'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'BBCAstrometrySession'

ssh_solver_config = 'an-bbc'
sitename = 'bbc'

SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
