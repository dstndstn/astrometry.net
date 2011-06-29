from settings_common import *

DATABASES['default']['NAME'] = 'an-staging'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'StagingAstrometrySession'

ssh_solver_config = 'an-nova'
sitename = 'staging'

