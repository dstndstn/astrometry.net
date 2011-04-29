from django.conf.urls.defaults import *

from astrometry.net2 import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
                       (r'^login$', 'astrometry.net2.views.login'),
                       (r'^login-openid', 'astrometry.net2.views.login_openid_done'),
                       (r'^logout', 'astrometry.net2.views.logout'),
                       (r'^submit_url', 'astrometry.net2.views.submit_url'),
                       (r'^submit_file', 'astrometry.net2.views.submit_file'),
                       (r'^submit_full', 'astrometry.net2.views.submit_full'),

                       (r'^mymedia/(?P<path>.*)$', 'django.views.static.serve',
                        {'document_root': settings.MEDIA_ROOT, 'show_indexes':True}),

					   # fallback
                       (r'', 'astrometry.net2.views.login'),


    # Example:
    # (r'^net2/', include('net2.foo.urls')),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # (r'^admin/', include(admin.site.urls)),
)
