from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
)

urlpatterns += patterns('astrometry.net.openid_views',
    url(r'^signin/?$', 'login_begin', name='openid-login'),
    url(r'^signout/?$', 'logout', name='openid-logout'),
    url(r'^complete/?$', 'login_complete', name='openid-complete'),
    url(r'^logo.gif$', 'logo', name='openid-logo'),
)

jobpattern = r'[0-9-]+'
subpattern = r'[0-9-]+'
imagepattern = r'[0-9-]+'
idpattern = r'[0-9-]+'

urlpatterns += patterns('astrometry.net.views.submission',
    (r'^upload/?$', 'upload_file'),
    (r'^status/(?P<subid>' + subpattern + r')/?', 'status'),
    (r'^submissions/(?P<user_id>' + idpattern + r')/?$', 'index'),
)

urlpatterns += patterns('astrometry.net.views.user',
    (r'^dashboard/?$', 'dashboard'),
    #(r'^dashboard/apikey/?$', 'get_api_key'),  # made redundant by inclusion of api key in dashboard profile
    (r'^dashboard/submissions/?$', 'dashboard_submissions'),
    (r'^dashboard/images/?', 'dashboard_user_images'),
    (r'^dashboard/profile/?$', 'dashboard_profile'),
    (r'^dashboard/profile/save/?$', 'save_profile'),
    (r'^users/?$', 'index'),
    (r'^users/(?P<user_id>' + idpattern + r')/?$', 'public_profile'),
)

urlpatterns += patterns('astrometry.net.views.image',
    (r'^annotated_(?P<size>full|display)/(?P<jobid>' + jobpattern + r')/?', 'annotated_image'),
    (r'^user_images/?$', 'index'),
    (r'^user_images/recent/?$', 'index_recent'),
    (r'^user_images/all/?$', 'index_all'),
    (r'^user_images/by_user/?$', 'index_by_user'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/?', 'user_image'),
    (r'^image/(?P<id>' + imagepattern + r')/?', 'serve_image'),
    (r'^images/(?P<category>\w+)/(?P<id>' + idpattern + r')/?', 'image_set'),
    (r'^allsky_plot/(?P<calid>' + idpattern + r')/?', 'onthesky_image'),
    (r'^sky_plot1/(?P<calid>' + idpattern + r')/?', 'onthesky_zoom1_image'),
    (r'^sky_plot2/(?P<calid>' + idpattern + r')/?', 'onthesky_zoom2_image'),
    (r'^sdss_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?', 'sdss_image'),
    (r'^galex_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?', 'galex_image'),
)

urlpatterns += patterns('astrometry.net.views.comment',
    (r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/comments/new/?', 'new'),
    (r'^comments/(?P<comment_id>' + idpattern + r')/delete/?', 'delete'),
)

urlpatterns += patterns('astrometry.net.api',
                        (r'^api/login/?', 'api_login'),
                        (r'^api/upload/?', 'api_upload'),
                        (r'^api/url_upload/?', 'url_upload'),
                        (r'^api/sdss_image_for_wcs/?', 'api_sdss_image_for_wcs'),
                        (r'^api/galex_image_for_wcs/?', 'api_galex_image_for_wcs'),
                        #(r'^api/logout/?', 'logout'),
)

# static file serving in development
if settings.DEBUG:
    urlpatterns += patterns('',
        (r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.STATICFILES_DIRS[0]}),
    )

# fallback
urlpatterns += patterns('astrometry.net.views.user',
                        (r'', 'dashboard'),
                        )
