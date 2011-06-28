# settings_nova.py
from settings_common import *

DATABASES['default']['NAME'] = 'an-nova'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'NovaAstrometrySession'

ssh_solver_config = 'an-nova'
sitename = 'nova'

