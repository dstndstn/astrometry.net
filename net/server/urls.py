from django.conf.urls.defaults import *

urlpatterns = patterns('',
                       (r'^server/?$', 'astrometry.net.server.views.summary'),
                       (r'^server/input/$', 'astrometry.net.server.views.get_input'),
                       (r'^server/results/$', 'astrometry.net.server.views.set_results'),
                       # fake
                       (r'^anmedia/', 'astrometry.net.media'),
                       )
