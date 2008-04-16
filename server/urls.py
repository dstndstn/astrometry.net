from django.conf.urls.defaults import *

urlpatterns = patterns('',
                       (r'^server/?$', 'astrometry.server.views.summary'),
                       # fake
                       (r'^anmedia/', 'astrometry.net.media'),
                       )
