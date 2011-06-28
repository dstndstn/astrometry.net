# settings_supernova.py
from settings_common import *

DATABASES['default']['NAME'] = 'an-supernova'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'SupernovaAstrometrySession'

# Used in process_submissions
ssh_solver_config = 'an-supernova'
sitename = 'supernova'

