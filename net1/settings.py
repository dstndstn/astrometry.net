import astrometry.net1.secrets.django_db as secrets

#GMAPS_API_KEY = 'ABQIAAAA7dWWcc9pB-GTzZE7CvT6SRTpFHQyuc9zMeXV0wfLqFiAr83b_xQ02hHbHnV4CjuIf4L-3WI_XATmBQ'

# edge.astrometry.net
#GMAPS_API_KEY = 'ABQIAAAA7dWWcc9pB-GTzZE7CvT6SRRd6OZDG7afgOT-qBD56qXrD_4sXBRoWqZoA0bluUpwPo-gQuBqRm5Tug'
# oven.cosmo.fas.nyu.edu
GMAPS_API_KEY = 'ABQIAAAAdOKbO45hSEoNCGlzTiew7BRWPaQfegoYWoxyhnGKpr3zYcSQBxRo-Gk2drjLibynnK4VOeUcCehGxA'

GMAPS_HOSTS = ['edge%i.astrometry.net' % i for i in [1,2,3,4,5]]

W3C_VALIDATOR_URL = 'http://oven.cosmo.fas.nyu.edu:8888/w3c-markup-validator/check'

#UPLOADER_URL = '/test/uploader'
UPLOADER_URL = '/uploader'

UPLOAD_DIR = '/data2/TEMP-test'

#BASEDIR = '/home/gmaps/test/'
BASEDIR = '/home/dstn/astrometry/src/'
DATADIR = '/home/dstn/test/'

LOG_DIR = BASEDIR + 'astrometry/net1/log/'
ANDIR = BASEDIR + 'astrometry/'
UTIL_DIR = ANDIR + 'util/'
BLIND_DIR = ANDIR + 'blind/'
WEB_DIR = ANDIR + 'net1/'
JOB_QUEUE_DIR = DATADIR + 'job-queue/'

BASE_URL = 'http://oven.cosmo.fas.nyu.edu:9000/'

# for astrometry.net.server
MAIN_SERVER = 'http://oven.cosmo.fas.nyu.edu:8888'
SERVER_LOGFILE = LOG_DIR + 'server.log'

#BACKEND_CONFIG = '/home/dstn/go/django/backend-%s.cfg'
BACKEND_CONFIG = '/home/gmaps/shard-backend.cfg'

LOGFILE = LOG_DIR + 'django.log'
PORTAL_LOGFILE = LOG_DIR + 'portal.log'
VO_LOGFILE = LOG_DIR + 'vo.log'

HENRY_DRAPER_CAT = ANDIR + 'net1/hd.fits'
TYCHO_MKDT = ANDIR + 'net1/tycho.mkdt.fits'

TILERENDER = ANDIR + 'render/tilerender'
SIPSO = ANDIR + 'util/_sip.so'

CACHEDIR = DATADIR + 'tilecache/'
RENDERCACHEDIR = DATADIR + 'rendercache/'
TEMPDIR = '/tmp'

#SITE_NAME = 'test'
SITE_NAME = 'edge'
JOB_DIR = DATADIR + 'web-data/'
#FIELD_DIR = '/home/gmaps/test/web-data/fields'
FIELD_DIR = DATADIR + 'fields'

APPEND_SLASH=False

DEBUG = True
TEMPLATE_DEBUG = DEBUG

ADMINS = (
    # ('Your Name', 'your_email@domain.com'),
)

ACCOUNT_ADMIN = 'dstn@cs.toronto.edu'

SESSION_COOKIE_NAME = 'AstrometryTestSession'

MANAGERS = ADMINS

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
		# Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
        'NAME': 'an-test',
        'USER': secrets.DATABASE_USER,
        'PASSWORD': secrets.DATABASE_PASSWORD,
        'HOST': secrets.DATABASE_HOST,
        'PORT': secrets.DATABASE_PORT,
		}
}

## --> see astrometry/net/fixtures/site.json
# Load with: python manage.py loaddata fixtures/site.json
SITE_ID = 2

DEFAULT_FROM_EMAIL = 'Astrometry.net <alpha@astrometry.net>'

# Local time zone for this installation. Choices can be found here:
# http://www.postgresql.org/docs/8.1/static/datetime-keywords.html#DATETIME-TIMEZONE-SET-TABLE
# although not all variations may be possible on all operating systems.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'America/Toronto'

# Language code for this installation. All choices can be found here:
# http://www.w3.org/TR/REC-html40/struct/dirlang.html#langcodes
# http://blogs.law.harvard.edu/tech/stories/storyReader$15
LANGUAGE_CODE = 'en-us'

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# Absolute path to the directory that holds media.
MEDIA_ROOT = WEB_DIR + 'media/'

# URL that handles the media served from MEDIA_ROOT.
# Example: "http://media.lawrence.com"
#MEDIA_URL = ''

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/admin-media/'

# Make this unique, and don't share it with anybody.
SECRET_KEY = '*pc#4fb*(%4gvp1-5yq6a_s&=4!gnui9r*53d+!*&s0=(@_ida'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
	'django.template.loaders.filesystem.Loader',
	'django.template.loaders.app_directories.Loader',
)

TEMPLATE_CONTEXT_PROCESSORS = (
    # default set:
	'django.contrib.auth.context_processors.auth',
    'django.core.context_processors.debug',
    'django.core.context_processors.i18n',
    'django.core.context_processors.media',
    # added:
    'django.core.context_processors.request',
    )


MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    #'django.middleware.doc.XViewMiddleware',
	)

ROOT_URLCONF = 'astrometry.net1.root-urls-test'

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/job/newurl/'

TEMPLATE_DIRS = (
    WEB_DIR,
)

AUTH_PROFILE_MODULE = 'portal.userprofile'

INSTALLED_APPS = (
	'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
	'django.contrib.admin',
	'astrometry.net1.tile',
	'astrometry.net1.upload',
	'astrometry.net1.portal',
	#'astrometry.net1.vo',
    #'astrometry.net1.testbed',
)



