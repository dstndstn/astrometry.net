from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
)

urlpatterns += patterns('astrometry.net.openid_views',
    url(r'^login/?$', 'login_begin', name='openid-login'),
    url(r'^logout/?$', 'logout', name='openid-logout'),
    url(r'^complete/?$', 'login_complete', name='openid-complete'),
    url(r'^logo.gif$', 'logo', name='openid-logo'),
)

jobpattern = r'[0-9-]+'
subpattern = r'[0-9-]+'

urlpatterns += patterns('astrometry.net.views.submission',
    (r'^upload/?$', 'upload_file'),
    (r'^status/(?P<subid>' + subpattern + r')/?', 'status'),
)

urlpatterns += patterns('astrometry.net.views.user',
    (r'^dashboard/?$', 'dashboard'),
    (r'^apikey/?$', 'get_api_key'),
)

urlpatterns += patterns('astrometry.net.views.image',
    (r'^annotated/(?P<jobid>' + jobpattern + r')/?', 'annotated_image'),
    (r'^sdss_image/(?P<jobid>' + jobpattern + r')/?', 'sdss_image'),
    (r'^submitted_image/(?P<jobid>' + jobpattern + r')/?', 'submitted_image'),
)

urlpatterns += patterns('astrometry.net.api',
                        (r'^api/login/?', 'api_login'),
                        (r'^api/upload/?', 'api_upload'),
                        #(r'^api/logout/?', 'logout'),
)

# fallback
urlpatterns += patterns('astrometry.net.views.user',
                        (r'', 'dashboard'),
                        )
