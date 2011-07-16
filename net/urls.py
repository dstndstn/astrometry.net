from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
)

urlpatterns += patterns('astrometry.net.views.home',
    (r'^/?$', 'home'),
    (r'^support/?$', 'support'),
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
tagpattern = r'[\s|\S]+'

urlpatterns += patterns('astrometry.net.views.submission',
    (r'^upload/?$', 'upload_file'),
    (r'^status/(?P<subid>' + subpattern + r')/?', 'status'),
    (r'^joblog/(?P<jobid>' + jobpattern + r')/?', 'job_log_file'),
    (r'^joblog2/(?P<jobid>' + jobpattern + r')/?', 'job_log_file2'),
    (r'^submissions/(?P<user_id>' + idpattern + r')/?$', 'index'),
)

urlpatterns += patterns('astrometry.net.views.user',
    (r'^dashboard/?$', 'dashboard'),
    #(r'^dashboard/apikey/?$', 'get_api_key'),  # made redundant by inclusion of api key in dashboard profile
    (r'^dashboard/submissions/?$', 'dashboard_submissions'),
    (r'^dashboard/images/?', 'dashboard_user_images'),
    (r'^dashboard/albums/?', 'dashboard_albums'),
    (r'^dashboard/create_album/?', 'dashboard_create_album'),
    (r'^dashboard/profile/?$', 'dashboard_profile'),
    (r'^dashboard/profile/save/?$', 'save_profile'),
    (r'^users/?$', 'index'),
    (r'^users/(?P<user_id>' + idpattern + r')/?$', 'user_profile'),
    (r'^users/(?P<user_id>' + idpattern + r')/images/?$', 'user_images'),
    (r'^users/(?P<user_id>' + idpattern + r')/albums/?$', 'user_albums'),
    (r'^users/(?P<user_id>' + idpattern + r')/submissions/?$', 'user_submissions'),
)

urlpatterns += patterns('astrometry.net.views.image',
    (r'^annotated_(?P<size>full|display)/(?P<jobid>' + jobpattern + r')/?', 'annotated_image'),
    (r'^user_images/?$', 'index'),
    (r'^user_images/recent/?$', 'index_recent'),
    (r'^user_images/all/?$', 'index_all'),
    (r'^user_images/by_user/?$', 'index_by_user'),
    (r'^user_images/user/(?P<user_id>' + idpattern + r')/?$', 'index_user'),
    (r'^user_images/album/(?P<album_id>' + idpattern + r')/?$', 'index_album'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/hide/?$', 'hide'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/unhide/?$', 'unhide'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/?$', 'user_image'),
    (r'^user_images/search/?$', 'search'),
    (r'^image/(?P<id>' + imagepattern + r')/?', 'serve_image'),
    (r'^images/(?P<category>\w+)/(?P<id>' + idpattern + r')/?', 'image_set'),
    (r'^sky_plot/zoom(?P<zoom>[0-3])/(?P<calid>' + idpattern + r')/?', 'onthesky_image'),
    (r'^sdss_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?', 'sdss_image'),
    (r'^galex_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?', 'galex_image'),
    (r'^wcs_file/(?P<jobid>' + idpattern + r')/?', 'wcs_file'),
)

urlpatterns += patterns('astrometry.net.views.album',
    (r'^albums/(?P<album_id>' + idpattern + r')/delete/?', 'delete'),
    (r'^albums/(?P<album_id>' + idpattern + r')/?', 'album'),
    (r'^albums/new/?', 'new'),
)
urlpatterns += patterns('astrometry.net.views.tag',
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/tags/(?P<tag_id>' + tagpattern + r')/remove/?', 'remove_userimagetag'),
)


urlpatterns += patterns('astrometry.net.views.comment',
    (r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/comments/new/?', 'new'),
    (r'^comments/(?P<comment_id>' + idpattern + r')/delete/?', 'delete'),
)

psidpattern = r'[0-9-]+'

urlpatterns += patterns('astrometry.net.views.admin',
                        (r'^admin/procsub/(?P<psid>'+psidpattern + r')?', 'procsub'),
                        (r'^admin/?', 'index'),
                        )

urlpatterns += patterns('astrometry.net.api',
    (r'^api/login/?', 'api_login'),
    (r'^api/upload/?', 'api_upload'),
    (r'^api/url_upload/?', 'url_upload'),
    (r'^api/sdss_image_for_wcs/?', 'api_sdss_image_for_wcs'),
    (r'^api/galex_image_for_wcs/?', 'api_galex_image_for_wcs'),
    (r'^api/submission_images/?', 'api_submission_images'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/?$', 'job_status'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/calibration/?', 'calibration'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/tags/?', 'tags'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/machine_tags/?', 'machine_tags'),
    #(r'^api/logout/?', 'logout'),
)


# static file serving in development
if settings.DEBUG:
    urlpatterns += patterns('',
        (r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.STATICFILES_DIRS[0]}),
    )

# fallback
urlpatterns += patterns('astrometry.net.views.home',
                        (r'', 'home'),
                        )
