from __future__ import absolute_import
from .settings_common import *

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-abc'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'ABCAstrometrySession'

ssh_solver_config = 'an-abc'
sitename = 'abc'

SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
