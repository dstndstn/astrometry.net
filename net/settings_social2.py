import astrometry.net.appsecrets.auth as authsecrets

# The namespace for social authentication is horrible.

# This uses the Python Social Auth module,
#  http://python-social-auth-docs.readthedocs.io/en/latest/configuration/settings.html
# which was derived from the django-social-auth module.


SOCIAL_INSTALLED_APPS = (
    'social_django',
)

SOCIAL_AUTH_BACKENDS = (
    'social_core.backends.google.GoogleOAuth2',
    #'social_core.backends.twitter.TwitterOAuth',
    'social_core.backends.yahoo.YahooOAuth2',
    'social_core.backends.github.GithubOAuth2',
    'social_core.backends.flickr.FlickrOAuth',
    'social_core.backends.evernote.EvernoteOAuth',
    'social_core.backends.amazon.AmazonOAuth2',
    #'django.contrib.auth.backends.ModelBackend',
)

SOCIAL_TEMPLATE_CONTEXT_PROCESSORS = (
    'social_django.context_processors.backends',
    'social_django.context_processors.login_redirect',
)

# SOCIAL_MIGRATION = {
#     'default': 'social.apps.django_app.default.south_migrations'
#     }

# SOCIAL_AUTH_STRATEGY = 'social.strategies.django_strategy.DjangoStrategy'
# SOCIAL_AUTH_STORAGE = 'social.apps.django_app.default.models.DjangoStorage'

SOCIAL_AUTH_REDIRECT_IS_HTTPS = True

SOCIAL_AUTH_GOOGLE_OAUTH2_KEY    = authsecrets.google.key
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = authsecrets.google.secret
SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = ['openid', 'email']

# dstn can't figure out how to get user's email addr from flickr.  Argh!
SOCIAL_AUTH_FLICKR_KEY    = authsecrets.flickr.key
SOCIAL_AUTH_FLICKR_SECRET = authsecrets.flickr.secret
SOCIAL_AUTH_FLICKR_SCOPE = ['openid', 'email']
### I was getting a Flickr SSL verification error...
SOCIAL_AUTH_FLICKR_VERIFY_SSL = False

github_secrets = authsecrets.githubs

# SOCIAL_AUTH_GITHUB_KEY    = authsecrets.githubs[sitename].key
# SOCIAL_AUTH_GITHUB_SECRET = authsecrets.githubs[sitename].secret
# #SOCIAL_AUTH_GITHUB_SCOPE = ['openid', 'email']
SOCIAL_AUTH_GITHUB_SCOPE = ['user:email']

# dstn can't figure out how to get user's email addr from twitter.  Argh!
# https://twittercommunity.com/t/how-to-get-email-from-twitter-user-using-oauthtokens/558/74
SOCIAL_AUTH_TWITTER_KEY    = authsecrets.twitter.key
SOCIAL_AUTH_TWITTER_SECRET = authsecrets.twitter.secret
SOCIAL_AUTH_TWITTER_SCOPE = ['email']
#SOCIAL_AUTH_TWITTER_SCOPE = ['user:email']

# Key not working.... keep getting 401 auth req'd, with message oauth_problem=consumer_key_rejected
# SOCIAL_AUTH_YAHOO_OAUTH_KEY    = authsecrets.yahoo.key
# SOCIAL_AUTH_YAHOO_OAUTH_SECRET = authsecrets.yahoo.secret
# SOCIAL_AUTH_YAHOO_OAUTH_VERIFY_SSL = False

SOCIAL_AUTH_YAHOO_OAUTH2_KEY    = authsecrets.yahoo.key
SOCIAL_AUTH_YAHOO_OAUTH2_SECRET = authsecrets.yahoo.secret

SOCIAL_AUTH_RAISE_EXCEPTIONS = True

#SOCIAL_AUTH_USERNAME_IS_FULL_EMAIL = True

SOCIAL_AUTH_EVERNOTE_KEY = authsecrets.evernote.key
SOCIAL_AUTH_EVERNOTE_SECRET = authsecrets.evernote.secret

SOCIAL_AUTH_AMAZON_KEY = authsecrets.amazon.key
SOCIAL_AUTH_AMAZON_SECRET = authsecrets.amazon.secret
