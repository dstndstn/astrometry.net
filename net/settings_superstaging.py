# settings_supernova.py
from settings_common import *

DATABASES['default']['NAME'] = 'an-superstaging'

LOGGING['loggers']['django.request']['level'] = 'WARN'
#LOGGING['loggers']['django.db']['level'] = 'INFO'

SESSION_COOKIE_NAME = 'SuperstagingAstrometrySession'

# Used in process_submissions
ssh_solver_config = 'an-superstaging'
sitename = 'superstaging'

