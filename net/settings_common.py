import os
import astrometry.net
import astrometry.net.appsecrets.django_db as dbsecrets
import astrometry.net.appsecrets.auth as authsecrets
import astrometry.net.appsecrets.django as djsecrets
from astrometry.net.util import dict_pack

ALLOWED_HOSTS = ['astro.cs.toronto.edu', 'nova2.astrometry.net', 'localhost',
                 'supernova.astrometry.net', 'nova.astrometry.net',
                 'supernova3.astrometry.net',
                 'supernova4.astrometry.net',
                 'supernova5.astrometry.net',
                 'supernova6.astrometry.net',
                 'nova3.astrometry.net',
                 'nova4.astrometry.net',
                 'nova5.astrometry.net',
                 'nova6.astrometry.net',
                 'testserver',
                 'ccnova.astrometry.net',
]

MULTI_HOSTS = []

# broiler
#WCS2KML = '/usr/local/wcs2kml/bin/wcs2kml'
# bbq
WCS2KML = '/usr/local/bin/wcs2kml'

ENABLE_SOCIAL=False
ENABLE_SOCIAL2=False

os.environ['MPLCONFIGDIR'] = '/home/nova/.config/matplotlib'

DATE_FORMAT = 'Y-m-d'
DATETIME_FORMAT = 'Y-m-d\TH:i:s\Z'
TIME_FORMAT = 'H:i:s\Z'

WEB_DIR = os.path.realpath(os.path.dirname(astrometry.net.__file__)) + '/'

# Catalogs
CAT_DIR = os.path.join(os.path.dirname(os.path.dirname(WEB_DIR)),
                       'catalogs')

SDSS_TILE_DIR = os.path.join(WEB_DIR, 'sdss-tiles')
GALEX_JPEG_DIR = os.path.join(WEB_DIR, 'GALEX-jpegs')
#HENRY_DRAPER_CAT = os.path.join(WEB_DIR, 'hd.fits')
#HENRY_DRAPER_CAT = '/data1/catalogs-fits/HD/hd2.fits'
HENRY_DRAPER_CAT = '/data2/nova/hd.fits'
TYCHO2_KD = '/data2/nova/tycho2.kd'
HIPPARCOS_CAT = '/data2/nova/hip.fits'

DATADIR = os.path.join(WEB_DIR, 'data', 'files')
JOBDIR = os.path.join(WEB_DIR, 'data', 'jobs')

ENHANCE_DIR = os.path.join(WEB_DIR, 'data', 'files', 'enhance')

TEMPDIR = '/tmp'

# The 'host' name in ~/.ssh/config for running the compute server.
ssh_solver_config = 'an-test'
# the "site" part of the directory name to put files in on the compute server
sitename = 'test'

# FIXME
BASEDIR = '/home/dstn/astrometry/src/'
LOG_DIR = BASEDIR + 'astrometry/net/log/'
ANDIR = BASEDIR + 'astrometry/'
UTIL_DIR = ANDIR + 'util/'
SOLVER_DIR = ANDIR + 'solver/'
JOB_QUEUE_DIR = DATADIR + 'job-queue/'

JOB_DIR = DATADIR + 'web-data/'
#FIELD_DIR = '/home/gmaps/test/web-data/fields'
FIELD_DIR = DATADIR + 'fields'

WSGI_APPLICATION = 'astrometry.net.wsgi.application'

LOGIN_URL = '/signin'
LOGIN_REDIRECT_URL = '/dashboard/'

# Social
# LOGIN_URL = '/login/'
# LOGIN_REDIRECT_URL = '/done/'
URL_PATH = ''


LOGFILE = LOG_DIR + 'django.log'
PORTAL_LOGFILE = LOG_DIR + 'portal.log'
VO_LOGFILE = LOG_DIR + 'vo.log'

SESSION_SERIALIZER = 'django.contrib.sessions.serializers.JSONSerializer'

# must match fixtures/initial_data.json
MACHINE_USERNAME = "an-machine"
ANONYMOUS_USERNAME = "anonymous"
DEFAULT_LICENSE_ID = 1

# Plotting
#TYCHO_MKDT = ANDIR + 'net/tycho.mkdt.fits'
#TILERENDER = ANDIR + 'render/tilerender'
#SIPSO = ANDIR + 'util/_sip.so'
#CACHEDIR = DATADIR + 'tilecache/'
#RENDERCACHEDIR = DATADIR + 'rendercache/'


SESSION_COOKIE_NAME = 'AstrometryTestSession'

DEBUG = True
ADMINS = ()

MANAGERS = ADMINS

DATABASES = {
    'default': {
        # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': '',
        'USER': dbsecrets.DATABASE_USER,
        'PASSWORD': dbsecrets.DATABASE_PASSWORD,
        'HOST': dbsecrets.DATABASE_HOST,
        'PORT': dbsecrets.DATABASE_PORT,
        }
}

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# On Unix systems, a value of None will cause Django to use the same
# timezone as the operating system.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'UTC'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale
USE_L10N = False

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/media/"
# Absolute path to the directory that holds media.
MEDIA_ROOT = WEB_DIR + 'media/'

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://media.lawrence.com/media/", "http://example.com/media/"
MEDIA_URL = ''

# Absolute path to the directory static files should be collected to.
# Don't put anything in this directory yourself; store your static files
# in apps' "static/" subdirectories and in STATICFILES_DIRS.
# Example: "/home/media/media.lawrence.com/static/"
STATIC_ROOT = ''

# URL prefix for static files.
# Example: "http://media.lawrence.com/static/"
STATIC_URL = '/static/'

# URL prefix for admin static files -- CSS, JavaScript and images.
# Make sure to use a trailing slash.
# Examples: "http://foo.com/static/admin/", "/static/admin/".
ADMIN_MEDIA_PREFIX = '/static/admin/'

# Additional locations of static files
STATICFILES_DIRS = (
    os.path.join(WEB_DIR, 'static'),
)

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    # 'django.contrib.staticfiles.finders.DefaultStorageFinder',
)

# Make this unique, and don't share it with anybody.
SECRET_KEY = djsecrets.DJANGO_SECRET_KEY

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            # TEMPLATE_DIRS
            os.path.join(WEB_DIR, 'templates') ,
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                # Insert your TEMPLATE_CONTEXT_PROCESSORS here or use this
                # list if you haven't customized them:
                'django.contrib.auth.context_processors.auth',
                'django.template.context_processors.request',
                'django.template.context_processors.debug',
                'django.template.context_processors.i18n',
                'django.template.context_processors.media',
                'django.template.context_processors.static',
                'django.template.context_processors.tz',
                'django.contrib.messages.context_processors.messages',

                'astrometry.net.models.context_user_profile',
            ],
        },
    },
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'astrometry.net.tempfile_middleware.TempfileMiddleware',
]

ROOT_URLCONF = 'astrometry.net.urls'

ALLOWED_INCLUDE_ROOTS = (
    os.path.join(WEB_DIR, 'templates'),
)

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'astrometry.net.app.AstrometryNetConfig',
)

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
)

AUTH_PROFILE_MODULE = 'net.UserProfile'

SOUTH_MIGRATION_MODULES = {
}

MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'

# A sample logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        }
    },
    'formatters': {
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
                },
        'simple': {
            'format': '%(levelname)s %(message)s'
                },
    },
    'handlers': {
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler'
        },
        'null': {
            'level':'DEBUG',
            #'class':'django.utils.log.NullHandler',
            'class': 'logging.NullHandler',
        },
        'console':{
            'level':'DEBUG',
            'class':'logging.StreamHandler',
            'formatter': 'simple',
            'filters': ['require_debug_true'],
        },
    },
    'loggers': {
        'django.db': {                                                                                                                      
            'handlers': ['console'],                                                                                                        
            'level': 'INFO',
            #'level': 'DEBUG',
            'propagate': True,                                                                                                              
            },
        'django.request': {
            'handlers': ['console'],
            #'level': 'DEBUG',
            'level': 'INFO',
            'propagate': True,
        },
        'astrometry': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    }
}



