import astrometry.net.secrets.auth as authsecrets

SOCIAL_TEMPLATE_CONTEXT_PROCESSORS = (
    'social.apps.django_app.context_processors.backends',
    'social.apps.django_app.context_processors.login_redirect',
# 'social.apps.django_app.context_processors.social_auth_by_type_backends',
)

SOCIAL_INSTALLED_APPS = (
#'django_openid_auth',
    'social.apps.django_app.default',
    # https://docs.djangoproject.com/en/1.7/topics/migrations/#upgrading-from-south
    #'south',
)
SOCIAL_AUTH_BACKENDS = (
    #'social.backends.open_id.OpenIdAuth',
    #'social.backends.google.GoogleOpenId',
    'social.backends.google.GoogleOAuth2',
    #'social.backends.google.GoogleOAuth',
    'social.backends.twitter.TwitterOAuth',
    'social.backends.yahoo.YahooOpenId',
    #'social.backends.stripe.StripeBackend',
    #'social.backends.steam.SteamBackend',
    #'social.backends.reddit.RedditBackend',
    #'social.backends.amazon.AmazonBackend',
    #'social.backends.browserid.BrowserIDBackend',
    #'social.backends.contrib.linkedin.LinkedinBackend',
    #'social.backends.contrib.skyrock.SkyrockBackend',
    'social.backends.flickr.FlickrOAuth',
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
)
SOCIAL_MIGRATION = {
    'default': 'social.apps.django_app.default.south_migrations'
    }


# SOCIAL_AUTH_LOGIN_REDIRECT_URL = '/signedin/'
# SOCIAL_AUTH_LOGIN_ERROR_URL = '/error/'
# SOCIAL_AUTH_LOGIN_URL = '/'
# SOCIAL_AUTH_NEW_USER_REDIRECT_URL = '/newuser/'
# SOCIAL_AUTH_NEW_ASSOCIATION_REDIRECT_URL = '/newassoc/'
# SOCIAL_AUTH_DISCONNECT_REDIRECT_URL = '/'
# SOCIAL_AUTH_INACTIVE_USER_URL = '/inactive-user/'

#SOCIAL_AUTH_USER_MODEL = 'django.contrib.auth.models.User'
#SOCIAL_AUTH_USER_MODEL = 'net.MyUser'

SOCIAL_AUTH_STRATEGY = 'social.strategies.django_strategy.DjangoStrategy'
SOCIAL_AUTH_STORAGE = 'social.apps.django_app.default.models.DjangoStorage'

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

SOCIAL_AUTH_RAISE_EXCEPTIONS = True

#SOCIAL_AUTH_USERNAME_IS_FULL_EMAIL = True

# SOCIAL_AUTH_PIPELINE = (
#     'net.views.home.load_user',
#     # Get the information we can about the user and return it in a simple
#     # format to create the user instance later. On some cases the details are
#     # already part of the auth response from the provider, but sometimes this
#     # could hit a provider API.
#     'social.pipeline.social_auth.social_details',
#     # Get the social uid from whichever service we're authing thru. The uid is
#     # the unique identifier of the given user in the provider.
#     'social.pipeline.social_auth.social_uid',
#     # Verifies that the current auth process is valid within the current
#     # project, this is were emails and domains whitelists are applied (if
#     # defined).
#     'social.pipeline.social_auth.auth_allowed',
#     # Checks if the current social-account is already associated in the site.
#     'social.pipeline.social_auth.social_user',
#     #'net.views.home.pre_get_username',
#     # Make up a username for this person, appends a random string at the end if
#     # there's any collision.
#     'social.pipeline.user.get_username',
#     #'net.views.home.post_get_username',
#     # Create a user account if we haven't found one yet.
#     'social.pipeline.user.create_user',
#     #'net.views.home.post_create_user',
#     'social.pipeline.social_auth.associate_user',
#     'social.pipeline.social_auth.load_extra_data',
#     'social.pipeline.user.user_details',
#     'net.views.home.post_auth',
# )
