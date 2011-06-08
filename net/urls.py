from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
	(r'^openid/', include('django_openid_auth.urls')),
	(r'^logout/$', 'django.contrib.auth.views.logout'),
)

jobpattern = r'[0-9-]+'
subpattern = r'[0-9-]+'

urlpatterns += patterns('astrometry.net.views',
    (r'^dashboard/$', 'dashboard'),
    (r'^upload/$', 'upload_file'),
    (r'^status/(?P<subid>' + subpattern + r')', 'status'),
    (r'^annotated/(?P<jobid>' + jobpattern + r')', 'annotated_image'),
    (r'^apikey/$', 'get_api_key'),
)

urlpatterns += patterns('astrometry.net.api',
                        (r'^api/login', 'api_login'),
                        #(r'^api/logout', 'logout'),
)

# fallback
urlpatterns += patterns('astrometry.net.views',
                        (r'', 'dashboard'),
                        )
