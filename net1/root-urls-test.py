from django.conf.urls.defaults import *
from astrometry.net import settings

urlpatterns = patterns('',
					   (r'', include('astrometry.net1.urls')),
					   #(r'^easy-gmaps', 'astrometry.net1.portal.easy_gmaps.tile'),
					   )

