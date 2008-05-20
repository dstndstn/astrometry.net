from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.net.server.master',
                       (r'^/solve/$',  'solve'),
                       (r'^/cancel/$', 'cancel'),
                       )
