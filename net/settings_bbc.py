# settings_ison.py
from settings_common import *

TEMPDIR = '/data2/tmp'
DATABASES['default']['NAME'] = 'an-bbc'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'BBCAstrometrySession'

ssh_solver_config = 'an-bbc'
sitename = 'bbc'

