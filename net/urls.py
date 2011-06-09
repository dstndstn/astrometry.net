from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    (r'^logout/?$', 'django.contrib.auth.views.logout'),
)

urlpatterns += patterns('astrometry.net.openid_views',
    url(r'^login/?$', 'login_begin', name='openid-login'),
    url(r'^complete/?$', 'login_complete', name='openid-complete'),
    url(r'^logo.gif$', 'logo', name='openid-logo'),
)

jobpattern = r'[0-9-]+'
subpattern = r'[0-9-]+'

urlpatterns += patterns('astrometry.net.views',
    (r'^dashboard/?$', 'dashboard'),
    (r'^upload/?$', 'upload_file'),
    (r'^status/(?P<subid>' + subpattern + r')/?', 'status'),
    (r'^annotated/(?P<jobid>' + jobpattern + r')/?', 'annotated_image'),
    (r'^submitted_image/(?P<jobid>' + jobpattern + r')/?', 'submitted_image'),
    (r'^apikey/?$', 'get_api_key'),
)

urlpatterns += patterns('astrometry.net.api',
                        (r'^api/login/?', 'api_login'),
                        (r'^api/upload/?', 'api_upload'),
                        #(r'^api/logout/?', 'logout'),
)

# fallback
urlpatterns += patterns('astrometry.net.views',
                        (r'', 'dashboard'),
                        )
