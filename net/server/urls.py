from django.conf.urls.defaults import *

urlpatterns = patterns('',
                       (r'^server/?$', 'astrometry.server.views.summary'),
                       (r'^server/input/$', 'astrometry.server.views.get_input'),
                       (r'^server/results/$', 'astrometry.server.views.set_results'),
                       # fake
                       (r'^anmedia/', 'astrometry.net.media'),
                       )
