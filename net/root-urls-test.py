from django.conf.urls.defaults import *
from astrometry.net import settings

urlpatterns = patterns('',
					   (r'^test/', include('astrometry.net.urls')),
                       )
