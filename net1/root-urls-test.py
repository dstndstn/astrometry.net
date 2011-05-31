from django.conf.urls.defaults import *
from astrometry.net import settings

urlpatterns = patterns('',
					   (r'', include('astrometry.net.urls')),
					   (r'^easy-gmaps', 'astrometry.net.portal.easy_gmaps.tile'),
					   )

