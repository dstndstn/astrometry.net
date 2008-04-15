from django.conf.urls.defaults import *

urlpatterns = patterns('',
                       (r'^easy-gmaps/', 'proj.views.get_tile'),
                       )
