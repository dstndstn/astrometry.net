from django.conf.urls.defaults import *

urlpatterns = patterns('an.upload.views',
					   (r'^form/$', 'uploadform'),
					   (r'^progress_ajax/$', 'progress_ajax'),
					   (r'^xml/$', 'progress_xml'),
                       )
