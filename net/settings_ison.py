# settings_ison.py
from settings_common import *

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-ison'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'IsonAstrometrySession'

ssh_solver_config = 'an-ison'
sitename = 'ison'

