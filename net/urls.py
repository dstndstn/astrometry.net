from django.conf.urls import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
)

urlpatterns += patterns('astrometry.net.views.home',
    (r'^/?$', 'home'),
    url(r'^support/?$', 'support', name='support'),
    url(r'^api_help/?$', 'api_help', name='api-help'),
    url(r'^explore/?$', 'explore', name='explore'),
)

urlpatterns += patterns('astrometry.net.openid_views',
    url(r'^signin/?$', 'login_begin', name='openid-login'),
    url(r'^signout/?$', 'logout', name='openid-logout'),
    url(r'^complete/?$', 'login_complete', name='openid-complete'),
    url(r'^logo.gif$', 'logo', name='openid-logo'),
)

urlpatterns += patterns('astrometry.net.views.search',
    (r'^search/images/?$', 'images'),
    (r'^search/users/?$', 'users'),
)

jobpattern = r'[0-9-]+'
subpattern = r'[0-9-]+'
imagepattern = r'[0-9-]+'
idpattern = r'[0-9-]+'
tagpattern = r'[\s|\S]+'

urlpatterns += patterns('astrometry.net.views.submission',
    url(r'^upload/?$', 'upload_file', name='upload-file'),
    url(r'^status/(?P<subid>' + subpattern + r')/?', 'status',
        name='sub-status'),
    url(r'^joblog/(?P<jobid>' + jobpattern + r')/?', 'job_log_file',
        name='job-log-file-1'),
    url(r'^joblog2/(?P<jobid>' + jobpattern + r')/?', 'job_log_file2',
        name='job-log-file-2'),
    (r'^submissions/(?P<user_id>' + idpattern + r')/?$', 'index'),
)

urlpatterns += patterns('astrometry.net.views.user',
    url(r'^dashboard/?$', 'dashboard', name='dashboard'),
    #(r'^dashboard/apikey/?$', 'get_api_key'),  # made redundant by inclusion of api key in dashboard profile
    (r'^dashboard/submissions/?$', 'dashboard_submissions'),
    (r'^dashboard/images/?$', 'dashboard_user_images'),
    (r'^dashboard/albums/?$', 'dashboard_albums'),
    (r'^dashboard/create_album/?$', 'dashboard_create_album'),
    (r'^dashboard/profile/?$', 'dashboard_profile'),
    (r'^dashboard/profile/save/?$', 'save_profile'),
    url(r'^users/?$', 'index', name='users'),
    url(r'^users/(?P<user_id>' + idpattern + r')/?$', 'user_profile',
        name='user-profile'),
    (r'^users/(?P<user_id>' + idpattern + r')/images/?$', 'user_images'),
    (r'^users/(?P<user_id>' + idpattern + r')/albums/?$', 'user_albums'),
    (r'^users/(?P<user_id>' + idpattern + r')/submissions/?$', 'user_submissions'),
    (r'^users/autocomplete/?$', 'user_autocomplete'),
)

urlpatterns += patterns('astrometry.net.views.image',
    url(r'^annotated_(?P<size>full|display)/(?P<jobid>' + jobpattern + r')/?', 'annotated_image', name='annotated_image'),
    url(r'^user_images/?$', 'index', name='images'),
    url(r'^user_images/tag/?$', 'index_tag', name='images-tag'),
    url(r'^user_images/location/?$', 'index_location', name='images-location'),
    url(r'^user_images/nearby/(?P<user_image_id>' + idpattern + r')/?$', 'index_nearby',
     name='images-nearby'),
    (r'^user_images/recent/?$', 'index_recent'),
    (r'^user_images/all/?$', 'index_all'),
    (r'^user_images/by_user/?$', 'index_by_user'),
    (r'^user_images/user/(?P<user_id>' + idpattern + r')/?$', 'index_user'),
    (r'^user_images/album/(?P<album_id>' + idpattern + r')/?$', 'index_album'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/hide/?$', 'hide'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/unhide/?$', 'unhide'),
    url(r'^user_images/(?P<user_image_id>' + idpattern + r')/?$', 'user_image', name='user-image'),
    (r'^user_images/(?P<user_image_id>' + idpattern + r')/edit/?$', 'edit'),
    url(r'^user_images/search/?$', 'search', name='image-search'),
    url(r'^image/(?P<id>' + imagepattern + r')/?$', 'serve_image', name='serve_image'),
    (r'^images/(?P<category>\w+)/(?P<id>' + idpattern + r')/?$', 'image_set'),
    url(r'^sky_plot/zoom(?P<zoom>[0-3])/(?P<calid>' + idpattern + r')/?$', 'onthesky_image', name='onthesky_image'),
    url(r'^sdss_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?$', 'sdss_image', name='sdss_image'),
    url(r'^galex_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?$', 'galex_image', name='galex_image'),
    url(r'^red_green_image_(?P<size>full|display)/(?P<job_id>' + idpattern + r')/?$', 'red_green_image', name='red_green_image'),
    url(r'^extraction_image_(?P<size>full|display)/(?P<job_id>' + idpattern + r')/?$', 'extraction_image', name='extraction_image'),
    url(r'^wcs_file/(?P<jobid>' + idpattern + r')/?$', 'wcs_file', name='wcs-file'),
    url(r'^new_fits_file/(?P<jobid>' + idpattern + r')/?$', 'new_fits_file', name='new-fits-file'),
    url(r'^kml_file/(?P<jobid>' + idpattern + r')/?$', 'kml_file', name='kml-file'),
    url(r'^rdls_file/(?P<jobid>' + idpattern + r')/?$', 'rdls_file', name='rdls-file'),
    url(r'^axy_file/(?P<jobid>' + idpattern + r')/?$', 'axy_file', name='axy-file'),
    url(r'^corr_file/(?P<jobid>' + idpattern + r')/?$', 'corr_file', name='corr-file'),
)

urlpatterns += patterns('astrometry.net.views.enhance',
    (r'^enhance_ui/(?P<user_image_id>' + idpattern + r')/?$', 'enhanced_ui'),
    url(r'^enhanced_image_(?P<size>full|display)/(?P<job_id>' + idpattern + r')/?$', 'enhanced_image', name='enhanced_image'),
)

urlpatterns += patterns('astrometry.net.views.album',
    (r'^albums/(?P<album_id>' + idpattern + r')/?$', 'album'),
    (r'^albums/(?P<album_id>' + idpattern + r')/delete/?$', 'delete'),
    (r'^albums/(?P<album_id>' + idpattern + r')/edit/?$', 'edit'),
    (r'^albums/new/?$', 'new'),
)
urlpatterns += patterns('astrometry.net.views.tag',
    url(r'^tags/?$', 'index', name='tags'),
    (r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/tags/(?P<tag_id>' + tagpattern + r')/delete/?$', 'delete'),
    (r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/tags/new/?$', 'new'),
    (r'^tags/autocomplete/?$', 'tag_autocomplete'),
)
urlpatterns += patterns('astrometry.net.views.flag',
    (r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/flags/update/?$', 'update_flags'),
)
urlpatterns += patterns('astrometry.net.views.comment',
    (r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/comments/new/?$', 'new'),
    (r'^comments/(?P<comment_id>' + idpattern + r')/delete/?$', 'delete'),
)
urlpatterns += patterns('astrometry.net.views.license',
    (r'^(?P<licensable_type>\w+)/(?P<licensable_id>' + idpattern + r')/license/edit/?$', 'edit'),
)

psidpattern = r'[0-9-]+'

urlpatterns += patterns('astrometry.net.views.admin',
                        (r'^admin/procsub/(?P<psid>'+psidpattern + r')?$', 'procsub'),
                        (r'^admin/?', 'index'),
                        )

urlpatterns += patterns('astrometry.net.api',
    (r'^api/login/?$', 'api_login'),
    (r'^api/upload/?$', 'api_upload'),
    (r'^api/url_upload/?$', 'url_upload'),
    (r'^api/sdss_image_for_wcs/?$', 'api_sdss_image_for_wcs'),
    (r'^api/galex_image_for_wcs/?$', 'api_galex_image_for_wcs'),
    (r'^api/submission_images/?$', 'api_submission_images'),
    (r'^api/submissions/(?P<sub_id>' + idpattern + r')/?$', 'submission_status'),
	(r'^api/myjobs/', 'myjobs'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/?$', 'job_status'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/calibration/?$', 'calibration'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/tags/?$', 'tags'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/machine_tags/?$', 'machine_tags'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/objects_in_field/?$', 'objects_in_field'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/annotations/?$', 'annotations_in_field'),
    (r'^api/jobs/(?P<job_id>' + idpattern + r')/info/?$', 'job_info'),
    (r'^api/jobs_by_tag/?$', 'jobs_by_tag'),
    #(r'^api/logout/?', 'logout'),
)


# static file serving in development
if settings.DEBUG:
    urlpatterns += patterns('',
        (r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.STATICFILES_DIRS[0]}),
    )

# fallback
urlpatterns += patterns('astrometry.net.views.home',
                        url(r'', 'home', name='home'),
                        )
