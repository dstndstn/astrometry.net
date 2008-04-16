from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.web.upload.views',
					   (r'^form/$', 'uploadform'),
					   (r'^progress_ajax/$', 'progress_ajax'),
					   (r'^xml/$', 'progress_xml'),
                       )
