from __future__ import absolute_import
sitename = 'nova2'

from settings_common import *

# from settings_social import *
# ENABLE_SOCIAL = True
# SOUTH_MIGRATION_MODULES.update(SOCIAL_MIGRATION)
# TEMPLATE_CONTEXT_PROCESSORS += SOCIAL_TEMPLATE_CONTEXT_PROCESSORS
# INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
# AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS

from settings_social2 import *
ENABLE_SOCIAL2 = True
# SOUTH_MIGRATION_MODULES.update(SOCIAL_MIGRATION)
#TEMPLATE_CONTEXT_PROCESSORS += SOCIAL_TEMPLATE_CONTEXT_PROCESSORS
INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS

TEMPLATES[0]['OPTIONS']['context_processors'].extend(SOCIAL_TEMPLATE_CONTEXT_PROCESSORS)

sitename = 'nova2'

TEMPDIR = '/data1/tmp'
DATABASES['default']['NAME'] = 'an-nova2'

LOGGING['loggers']['django.request']['level'] = 'WARN'

SESSION_COOKIE_NAME = 'Nova2AstrometrySession'

ssh_solver_config = 'an-nova2'

try:
    SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
    SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
except:
    SOCIAL_AUTH_GITHUB_KEY    = None
    SOCIAL_AUTH_GITHUB_SECRET = None
    
