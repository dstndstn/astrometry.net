from settings_common import *

DATABASES['default']['NAME'] = 'an-nova'

LOGGING['loggers']['django.request']['level'] = 'WARN'

ssh_solver_config = 'an-nova'
