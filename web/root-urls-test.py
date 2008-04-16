from django.conf.urls.defaults import *
from astrometry.web import settings

urlpatterns = patterns('',
					   (r'^test/', include('astrometry.web.urls')),
                       )
