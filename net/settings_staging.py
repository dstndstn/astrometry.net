from settings_common import *

from settings_social import *
ENABLE_SOCIAL = True

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-staging'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'StagingAstrometrySession'

ssh_solver_config = 'an-nova'
sitename = 'staging'

SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
