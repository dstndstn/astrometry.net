from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.web.tile.views',
					   (r'^get/$', 'get_tile'),
					   (r'^list/$', 'get_image_list'),
					   (r'^image/$', 'get_image'),
                       #(r'^/$', 'index'),
					   )
