from django.conf.urls.defaults import *
from an import settings

urlpatterns = patterns('',
					   (r'^', include('an.urls')),
                       )
