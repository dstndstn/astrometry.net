from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.net.server.shard',
                       (r'^/solve/$',  'solve'),
                       (r'^/cancel/$', 'cancel'),
                       (r'^/index/$', 'index'),
                       )
