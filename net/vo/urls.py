from django.conf.urls.defaults import *

urlpatterns = patterns('an.vo.views',
					   #(r'^siap(-pointed)?/$', 'siap'),
					   (r'^siap-pointed/$', 'siap_pointed'),
					   (r'^siap-pointed-html/$', 'siap_pointed_html'),
					   (r'^getimage/$', 'getimage'),
                       )
