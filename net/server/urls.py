from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.net.server.views',
                       (r'^/?$',        'summary'    ),
                       (r'^/input/$',   'get_input'  ),
                       (r'^/results/$', 'set_results'),
                       (r'^/test', 'test'),
                       )
