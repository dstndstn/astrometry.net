# settings_nova.py
from settings_common import *

sitename = 'nova'

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-nova'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'NovaAstrometrySession'

ssh_solver_config = 'an-nova'

try:
    SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
    SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
except:
    SOCIAL_AUTH_GITHUB_KEY    = None
    SOCIAL_AUTH_GITHUB_SECRET = None
    
