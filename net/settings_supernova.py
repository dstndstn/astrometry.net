from __future__ import absolute_import

sitename = 'supernova'

# http://django-debug-toolbar.readthedocs.io/en/stable/installation.html
#INTERNAL_IPS = ['38.104.158.162']

DEBUG_TOOLBAR_PANELS = [
        'debug_toolbar.panels.versions.VersionsPanel',
        'debug_toolbar.panels.timer.TimerPanel',
        'debug_toolbar.panels.settings.SettingsPanel',
        'debug_toolbar.panels.headers.HeadersPanel',
        'debug_toolbar.panels.request.RequestPanel',
        'debug_toolbar.panels.sql.SQLPanel',
        'debug_toolbar.panels.staticfiles.StaticFilesPanel',
        'debug_toolbar.panels.templates.TemplatesPanel',
        'debug_toolbar.panels.cache.CachePanel',
        'debug_toolbar.panels.signals.SignalsPanel',
        'debug_toolbar.panels.logging.LoggingPanel',
        'debug_toolbar.panels.redirects.RedirectsPanel',
    ]

def always_show(*args, **kwargs):
    return True

DEBUG_TOOLBAR_CONFIG = dict(
    SHOW_TOOLBAR_CALLBACK = always_show,
    RENDER_PANELS = True,
    )

from settings_common import *

# from settings_social import *
# ENABLE_SOCIAL = True
# SOUTH_MIGRATION_MODULES.update(SOCIAL_MIGRATION)
# TEMPLATE_CONTEXT_PROCESSORS += SOCIAL_TEMPLATE_CONTEXT_PROCESSORS
# INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
# AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS

from astrometry.net.settings_social2 import *
ENABLE_SOCIAL2 = True
INSTALLED_APPS += SOCIAL_INSTALLED_APPS 
AUTHENTICATION_BACKENDS = SOCIAL_AUTH_BACKENDS + AUTHENTICATION_BACKENDS
TEMPLATES[0]['OPTIONS']['context_processors'].extend(SOCIAL_TEMPLATE_CONTEXT_PROCESSORS)

#INSTALLED_APPS = list(INSTALLED_APPS) + ['social.apps.django_app.default']

#SOCIAL_AUTH_LOGIN_REDIRECT_URL = 'http://supernova.astrometry.net/complete/'
#SOCIAL_AUTH_LOGIN_COMPLETE_URL = 'http://supernova.astrometry.net/complete/'
#SOCIAL_AUTH_SANITIZE_REDIRECTS = False
USE_X_FORWARDED_HOST = True

SITE_ID = 42

# ABSOLUTE_URL_OVERRIDES = {
#     'social_django.
# }

sitename = 'supernova'

DATABASES['default']['NAME'] = 'an-supernova'

LOGGING['loggers']['django.request']['level'] = 'WARN'
#LOGGING['loggers']['django.db']['level'] = 'INFO'

SESSION_COOKIE_NAME = 'SupernovaAstrometrySession'

# Used in process_submissions
ssh_solver_config = 'an-supernova'

try:
    SOCIAL_AUTH_GITHUB_KEY    = github_secrets[sitename].key
    SOCIAL_AUTH_GITHUB_SECRET = github_secrets[sitename].secret
except:
    SOCIAL_AUTH_GITHUB_KEY    = None
    SOCIAL_AUTH_GITHUB_SECRET = None
    
