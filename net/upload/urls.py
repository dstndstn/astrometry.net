from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.net.upload.views',
					   (r'^form/$',          'uploadform'   ),
					   (r'^progress_ajax/$', 'progress_ajax'),
					   (r'^xml/$',           'progress_xml' ),
                       )
