
# Add authentication tokens here for third-party Sign-In options
# (the XXX, YYY strings are only approximately the correct lengths)

class Duck(object):
    pass

# https://console.developers.google.com/
google = Duck()
google.key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXX.apps.googleusercontent.com'
google.secret = 'YYYYYYYYYYYYYYYYYYYYYYYY'

# https://www.flickr.com/services/apps
flickr = Duck()
flickr.key = 'XXXXXXXXXXXXXXXXXXXXXXXXXX'
flickr.secret = 'YYYYYYYYYYYYYYY'

# https://github.com/settings/applications/
githubs = {}
github = Duck()
github.key = 'XXXXXXXXXXXXXXXXXX'
github.secret = 'YYYYYYYYYYYYYYYYYYYYYYYYYYYY'
githubs['my-site'] = github

# etc etc etc
