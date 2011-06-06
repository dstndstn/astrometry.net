from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = (patterns('',
    (r'^openid/', include('django_openid_auth.urls')),
    (r'^logout/$', 'django.contrib.auth.views.logout'),
						) +
			   patterns('astrometry.net.views',
	(r'^dashboard/$', 'dashboard'),
	(r'^apikey/$', 'get_api_key'),
						) +
			   patterns('astrometry.net.api',
						(r'^api/login', 'login'),
						(r'^api/logout', 'logout'),
						)
)

						
