import os
import astrometry.net
import astrometry.net.secrets.django_db as secrets
import astrometry.net.secrets.auth as authsecrets
from astrometry.net.util import dict_pack

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
BLIND_DIR = ANDIR + 'blind/'
JOB_QUEUE_DIR = DATADIR + 'job-queue/'

JOB_DIR = DATADIR + 'web-data/'
#FIELD_DIR = '/home/gmaps/test/web-data/fields'
FIELD_DIR = DATADIR + 'fields'

WSGI_APPLICATION = 'astrometry.net.wsgi.application'

LOGIN_URL = '/signin'
LOGIN_REDIRECT_URL = '/dashboard/'

LOGFILE = LOG_DIR + 'django.log'
PORTAL_LOGFILE = LOG_DIR + 'portal.log'
VO_LOGFILE = LOG_DIR + 'vo.log'

# http://stackoverflow.com/questions/20301338/django-openid-auth-typeerror-openid-yadis-manager-yadisservicemanager-object-is
SESSION_SERIALIZER = 'django.contrib.sessions.serializers.PickleSerializer'

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
TEMPLATE_DEBUG = DEBUG

ADMINS = (
    # ('Your Name', 'your_email@example.com'),
)

MANAGERS = ADMINS

DATABASES = {
    'default': {
                # Add 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': '',
        'USER': secrets.DATABASE_USER,
        'PASSWORD': secrets.DATABASE_PASSWORD,
        'HOST': secrets.DATABASE_HOST,
        'PORT': secrets.DATABASE_PORT,
                }
}

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# On Unix systems, a value of None will cause Django to use the same
# timezone as the operating system.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'America/Toronto'

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
#MEDIA_ROOT = ''
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
STATIC_ROOT = ''#os.path.join(WEB_DIR, 'static')

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
    # 'django.contrib.staticfiles.finders.AppDirectoriesFinder',
    # 'django.contrib.staticfiles.finders.DefaultStorageFinder',
)

# Make this unique, and don't share it with anybody.
SECRET_KEY = 'd_&$%*@=ttb$qu047w0_35g=t@9+brymn)_si787g*52x_9e%n'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
)

ROOT_URLCONF = 'urls'

TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    os.path.join(WEB_DIR, 'templates') ,
)

ALLOWED_INCLUDE_ROOTS = (
    os.path.join(WEB_DIR, 'templates'),
)

TEMPLATE_CONTEXT_PROCESSORS = (
    'django.core.context_processors.debug',
    'django.core.context_processors.i18n',
    'django.core.context_processors.media',
    'django.core.context_processors.static',
    'django.core.context_processors.request',
    'django.contrib.auth.context_processors.auth',
    'django.contrib.messages.context_processors.messages',

    'social.apps.django_app.context_processors.backends',
    'social.apps.django_app.context_processors.login_redirect',
    #'social.apps.django_app.context_processors.social_auth_by_type_backends',
)

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Uncomment the next line to enable the admin:
    # 'django.contrib.admin',
    # Uncomment the next line to enable admin documentation:
    # 'django.contrib.admindocs',
    'astrometry.net',
    #'django_openid_auth',
    'social.apps.django_app.default',
    # https://docs.djangoproject.com/en/1.7/topics/migrations/#upgrading-from-south
    #'south',
)


#SOCIAL_AUTH_
AUTHENTICATION_BACKENDS = (
    #'social.backends.open_id.OpenIdAuth',
    #'social.backends.google.GoogleOpenId',
    'social.backends.google.GoogleOAuth2',
    #'social.backends.google.GoogleOAuth',

    ####'social.backends.twitter.TwitterOAuth',

    #'social.backends.yahoo.YahooOpenId',
    #'social.backends.stripe.StripeBackend',
    #'social.backends.steam.SteamBackend',
    #'social.backends.reddit.RedditBackend',
    #'social.backends.amazon.AmazonBackend',
    #'social.backends.browserid.BrowserIDBackend',
    #'social.backends.contrib.linkedin.LinkedinBackend',
    #'social.backends.contrib.skyrock.SkyrockBackend',

    #####'social.backends.flickr.FlickrOAuth',

    #'social.backends.contrib.instagram.InstagramBackend',
    'social.backends.github.GithubOAuth2',
    #'social.backends.contrib.yandex.YandexBackend',
    #'social.backends.contrib.yandex.YandexOAuth2Backend',
    #'social.backends.contrib.yandex.YaruBackend',
    #'social.backends.contrib.disqus.DisqusBackend',

    #####'social.backends.yahoo.YahooOAuth',

    #'social.backends.contrib.foursquare.FoursquareBackend',
    #'social.backends.contrib.live.LiveBackend',
    #'social.backends.contrib.livejournal.LiveJournalBackend',
    #'social.backends.contrib.douban.DoubanBackend',
    #'social.backends.contrib.vk.VKOpenAPIBackend',
    #'social.backends.contrib.vk.VKOAuth2Backend',
    #'social.backends.contrib.odnoklassniki.OdnoklassnikiBackend',
    #'social.backends.contrib.odnoklassniki.OdnoklassnikiAppBackend',
    #'social.backends.contrib.mailru.MailruBackend',
    #'social.backends.contrib.dailymotion.DailymotionBackend',
    #'social.backends.contrib.shopify.ShopifyBackend',
    #'social.backends.contrib.exacttarget.ExactTargetBackend',
    #'social.backends.contrib.stocktwits.StocktwitsBackend',
    #'social.backends.contrib.behance.BehanceBackend',
    #'social.backends.contrib.readability.ReadabilityBackend',
    #'social.backends.contrib.fedora.FedoraBackend',
    'django.contrib.auth.backends.ModelBackend',
)
SOCIAL_AUTH_LOGIN_REDIRECT_URL = '/signedin/'
SOCIAL_AUTH_LOGIN_ERROR_URL = '/error/'
SOCIAL_AUTH_LOGIN_URL = '/'
SOCIAL_AUTH_NEW_USER_REDIRECT_URL = '/newuser/'
SOCIAL_AUTH_NEW_ASSOCIATION_REDIRECT_URL = '/newassoc/'
SOCIAL_AUTH_DISCONNECT_REDIRECT_URL = '/'
SOCIAL_AUTH_INACTIVE_USER_URL = '/inactive-user/'

#SOCIAL_AUTH_USER_MODEL = 'django.contrib.auth.models.User'
#SOCIAL_AUTH_USER_MODEL = 'net.MyUser'

AUTH_PROFILE_MODULE = 'net.UserProfile'

SOCIAL_AUTH_GOOGLE_OAUTH2_KEY    = authsecrets.google.key
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = authsecrets.google.secret
SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = ['openid', 'email']

# dstn can't figure out how to get user's email addr from flickr.  Argh!

# SOCIAL_AUTH_FLICKR_KEY    = authsecrets.flickr.key
# SOCIAL_AUTH_FLICKR_SECRET = authsecrets.flickr.secret
# SOCIAL_AUTH_FLICKR_SCOPE = ['openid', 'email']

### I was getting a Flickr SSL verification error...
# SOCIAL_AUTH_FLICKR_VERIFY_SSL = False

github_secrets = authsecrets.githubs

# SOCIAL_AUTH_GITHUB_KEY    = authsecrets.githubs[sitename].key
# SOCIAL_AUTH_GITHUB_SECRET = authsecrets.githubs[sitename].secret
# #SOCIAL_AUTH_GITHUB_SCOPE = ['openid', 'email']
# SOCIAL_AUTH_GITHUB_SCOPE = ['user:email']

# dstn can't figure out how to get user's email addr from twitter.  Argh!
# https://twittercommunity.com/t/how-to-get-email-from-twitter-user-using-oauthtokens/558/74

# SOCIAL_AUTH_TWITTER_KEY    = authsecrets.twitter.key
# SOCIAL_AUTH_TWITTER_SECRET = authsecrets.twitter.secret
# SOCIAL_AUTH_TWITTER_SCOPE = ['email']
# #SOCIAL_AUTH_TWITTER_SCOPE = ['user:email']

# Key not working.... keep getting 401 auth req'd, with message oauth_problem=consumer_key_rejected

# SOCIAL_AUTH_YAHOO_OAUTH_KEY    = authsecrets.yahoo.key
# SOCIAL_AUTH_YAHOO_OAUTH_SECRET = authsecrets.yahoo.secret
# SOCIAL_AUTH_YAHOO_OAUTH_VERIFY_SSL = False


SOCIAL_AUTH_RAISE_EXCEPTIONS = True

#SOCIAL_AUTH_USERNAME_IS_FULL_EMAIL = True

SOCIAL_AUTH_PIPELINE = (
    'net.views.home.load_user',

    # Get the information we can about the user and return it in a simple
    # format to create the user instance later. On some cases the details are
    # already part of the auth response from the provider, but sometimes this
    # could hit a provider API.
    'social.pipeline.social_auth.social_details',

    # Get the social uid from whichever service we're authing thru. The uid is
    # the unique identifier of the given user in the provider.
    'social.pipeline.social_auth.social_uid',

    # Verifies that the current auth process is valid within the current
    # project, this is were emails and domains whitelists are applied (if
    # defined).
    'social.pipeline.social_auth.auth_allowed',

    # Checks if the current social-account is already associated in the site.
    'social.pipeline.social_auth.social_user',

    #'net.views.home.pre_get_username',

    # Make up a username for this person, appends a random string at the end if
    # there's any collision.
    'social.pipeline.user.get_username',

    #'net.views.home.post_get_username',

    # Create a user account if we haven't found one yet.
    'social.pipeline.user.create_user',

    #'net.views.home.post_create_user',

    'social.pipeline.social_auth.associate_user',
    'social.pipeline.social_auth.load_extra_data',
    'social.pipeline.user.user_details',

    'net.views.home.post_auth',
)



SOUTH_MIGRATION_MODULES = {
    'default': 'social.apps.django_app.default.south_migrations'
}






# AUTHENTICATION_BACKENDS = (
#     'django_openid_auth.auth.OpenIDBackend',
#     'django.contrib.auth.backends.ModelBackend',
# )
# 
# OPENID_CREATE_USERS = True
# OPENID_UPDATE_DETAILS_FROM_SREG = True

# list of open id providers to allow users to log into the site with;
# any instance of username will be replaced with a username

#OPENID_PROVIDERS = dict_pack(
#    ('provider', 'url', 'suggestion'),
#    ( # provider choice data
#        ('Google','google.com/accounts/o8/id',''),               # works
#        ('Yahoo','yahoo.com',''),                                # works
#        ('AOL','openid.aol.com/username','@aol.com'),            # works
#      # ('Launchpad','launchpad.net/~username',''),              # untested
#      # ('WordPress','username.wordpress.com','.wordpress.com'), # didn't work
#    )
#)

MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'

# A sample logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
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
            'class':'django.utils.log.NullHandler',
        },
        'console':{
            'level':'DEBUG',
            'class':'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'loggers': {
        'django.db': {                                                                                                                      
            'handlers': ['console'],                                                                                                        
            'level': 'INFO',
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



