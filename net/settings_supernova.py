ENABLE_SOCIAL = True

from settings_common import *

DATABASES['default']['NAME'] = 'an-supernova'

LOGGING['loggers']['django.request']['level'] = 'WARN'
#LOGGING['loggers']['django.db']['level'] = 'INFO'

SESSION_COOKIE_NAME = 'SupernovaAstrometrySession'

# Used in process_submissions
ssh_solver_config = 'an-supernova'

sitename = 'supernova'

SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
