from django.urls import include, re_path
from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = []

from astrometry.net.views.home import home, support, api_help, explore, maintenance

def dos_error(req):
    from django.http import HttpResponse
    return HttpResponse(content='django', status=503)

urlpatterns.extend([
    re_path(r'^new_fits_file/6490673$', dos_error)
    ])

urlpatterns.extend([
    #re_path(r'', maintenance),
    re_path(r'^$', home),
    re_path(r'^maintenance/?$', maintenance),
    re_path(r'^support/?$', support, name='support'),
    re_path(r'^api_help/?$', api_help, name='api-help'),
    # re_path(r'^new_api_key/?$', 'new_api_key', name='new_api_key'),
    re_path(r'^explore/?$', explore, name='explore'),
])

if settings.ENABLE_SOCIAL:
    urlpatterns.append(
        re_path('', include('social.apps.django_app.urls', namespace='social'))
    )

if settings.ENABLE_SOCIAL2:
    urlpatterns.append(
        re_path('', include('social_django.urls', namespace='social'))
    )

from astrometry.net.views.home import signin, signout, signedin, newuser
urlpatterns.extend([
    re_path(r'^signin/?',   signin, name='signin'),
    re_path(r'^signout/?',  signout, name='signout'),
    re_path(r'^signedin/?', signedin),
    re_path(r'^newuser/?',  newuser),
])

from astrometry.net.views.search import images, users
urlpatterns.extend([
    re_path(r'^search/images/?$', images),
    re_path(r'^search/users/?$',  users),
])

jobpattern = r'[0-9-]+'
subpattern = r'[0-9-]+'
imagepattern = r'[0-9-]+'
idpattern = r'[0-9-]+'
tagpattern = r'[\s|\S]+'

from astrometry.net.views.human import ask_human, am_human, test_human, not_human, test_human_2, poison

urlpatterns.extend([
    re_path(r'^ask_human$', ask_human),
    re_path(r'^am_human$', am_human),
    re_path(r'^not_human$', not_human),
    re_path(r'^test$', test_human),
    re_path(r'^test2/(?P<user_image_id>' + idpattern + r')/?$', test_human_2),
    re_path(r'^poison/(?P<depth>' + idpattern + r')/(?P<num>' + idpattern + r')/?$',
            poison),
])

from astrometry.net.views.submission import upload_file, status, job_log_file, job_log_file2, index
urlpatterns.extend([
    re_path(r'^upload/?$', upload_file, name='upload-file'),
    re_path(r'^status/(?P<subid>' + subpattern + r')/?', status,
        name='submission_status'),
    re_path(r'^joblog/(?P<jobid>' + jobpattern + r')/?', job_log_file,
        name='job_log_file'),
    re_path(r'^joblog2/(?P<jobid>' + jobpattern + r')/?', job_log_file2,
        name='job_log_file_2'),
    re_path(r'^submissions/(?P<user_id>' + idpattern + r')/?$', index),
])

from astrometry.net.views.user import (
    index, dashboard, user_profile, dashboard_submissions, dashboard_user_images, dashboard_albums,
    dashboard_create_album, dashboard_profile, save_profile, user_images, user_albums,
    user_submissions, user_autocomplete)
urlpatterns.extend([
    re_path(r'^dashboard/?$', dashboard, name='dashboard'),
    #(r'^dashboard/apikey/?$', 'get_api_key'),  # made redundant by inclusion of api key in dashboard profile
    re_path(r'^dashboard/submissions/?$', dashboard_submissions, name='dashboard_submissions'),
    re_path(r'^dashboard/images/?$', dashboard_user_images, name='dashboard_user_images'),
    re_path(r'^dashboard/albums/?$', dashboard_albums, name='dashboard_albums'),
    re_path(r'^dashboard/create_album/?$', dashboard_create_album, name='dashboard_create_album'),
    re_path(r'^dashboard/profile/?$', dashboard_profile, name='dashboard_profile'),
    re_path(r'^dashboard/profile/save/?$', save_profile, name='save_profile'),
    re_path(r'^users/?$', index, name='users'),
    re_path(r'^users/(?P<user_id>' + idpattern + r')/?$', user_profile, name='user_profile'),
    re_path(r'^users/(?P<user_id>' + idpattern + r')/images/?$', user_images, name='user_images'),
    re_path(r'^users/(?P<user_id>' + idpattern + r')/albums/?$', user_albums, name='user_albums'),
    re_path(r'^users/(?P<user_id>' + idpattern + r')/submissions/?$', user_submissions, name='user_submissions'),
    re_path(r'^users/autocomplete/?$', user_autocomplete, name='user_autocomplete'),
])

from astrometry.net.views.image import (
    index, index_tag, annotated_image, grid_image, index_location, index_nearby, index_recent, index_all, index_by_user,
    index_user, index_album, hide, unhide, user_image, edit, search, serve_image, serve_thumbnail_image, image_set,
    onthesky_image, sdss_image, galex_image, unwise_image, legacysurvey_image, red_green_image, extraction_image, wcs_file, new_fits_file,
    kml_file, rdls_file, axy_file, image_rd_file, corr_file)
urlpatterns.extend([
    re_path(r'^annotated_(?P<size>full|display)/(?P<jobid>' + jobpattern + r')/?', annotated_image, name='annotated_image'),
    re_path(r'^grid_(?P<size>full|display)/(?P<jobid>' + jobpattern + r')/?', grid_image, name='grid_image'),
    re_path(r'^user_images/?$', index, name='images'),
    re_path(r'^user_images/tag/?$', index_tag, name='images-tag'),
    re_path(r'^user_images/location/?$', index_location, name='images-location'),
    re_path(r'^user_images/nearby/(?P<user_image_id>' + idpattern + r')/?$', index_nearby, name='images-nearby'),
    re_path(r'^user_images/recent/?$', index_recent),
    re_path(r'^user_images/all/?$', index_all),
    re_path(r'^user_images/by_user/?$', index_by_user),
    re_path(r'^user_images/user/(?P<user_id>' + idpattern + r')/?$', index_user),
    re_path(r'^user_images/album/(?P<album_id>' + idpattern + r')/?$', index_album),
    re_path(r'^user_images/(?P<user_image_id>' + idpattern + r')/hide/?$', hide),
    re_path(r'^user_images/(?P<user_image_id>' + idpattern + r')/unhide/?$', unhide),
    re_path(r'^user_images/(?P<user_image_id>' + idpattern + r')/?$', user_image, name='user_image'),
    re_path(r'^user_images/(?P<user_image_id>' + idpattern + r')/edit/?$', edit, name='image_edit'),
    re_path(r'^user_images/search/?$', search, name='image-search'),
    re_path(r'^image/(?P<id>' + imagepattern + r')/?$', serve_image, name='serve_image'),
    re_path(r'^thumbnail_of_image/(?P<id>' + imagepattern + r')/?$', serve_thumbnail_image, name='serve_thumbnail_image'),
    re_path(r'^images/(?P<category>\w+)/(?P<id>' + idpattern + r')/?$', image_set),
    re_path(r'^sky_plot/zoom(?P<zoom>[0-3])/(?P<calid>' + idpattern + r')/?$', onthesky_image, name='onthesky_image'),
    re_path(r'^sdss_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?$', sdss_image, name='sdss_image'),
    re_path(r'^galex_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?$', galex_image, name='galex_image'),
    re_path(r'^unwise_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?$', unwise_image, name='unwise_image'),
    re_path(r'^legacysurvey_image_(?P<size>full|display)/(?P<calid>' + idpattern + r')/?$', legacysurvey_image, name='legacysurvey_image'),
    re_path(r'^red_green_image_(?P<size>full|display)/(?P<job_id>' + idpattern + r')/?$', red_green_image, name='red_green_image'),
    re_path(r'^extraction_image_(?P<size>full|display)/(?P<job_id>' + idpattern + r')/?$', extraction_image, name='extraction_image'),
    re_path(r'^wcs_file/(?P<jobid>' + idpattern + r')/?$', wcs_file, name='wcs-file'),
    re_path(r'^new_fits_file/(?P<jobid>' + idpattern + r')/?$', new_fits_file, name='new-fits-file'),
    re_path(r'^kml_file/(?P<jobid>' + idpattern + r')/?$', kml_file, name='kml-file'),
    re_path(r'^rdls_file/(?P<jobid>' + idpattern + r')/?$', rdls_file, name='rdls-file'),
    re_path(r'^axy_file/(?P<jobid>' + idpattern + r')/?$', axy_file, name='axy-file'),
    re_path(r'^image_rd_file/(?P<jobid>' + idpattern + r')/?$', image_rd_file, name='image-rd-file'),
    re_path(r'^corr_file/(?P<jobid>' + idpattern + r')/?$', corr_file, name='corr-file'),
])
#     
#     urlpatterns += patterns('astrometry.net.views.enhance',
#         (r'^enhance_ui/(?P<user_image_id>' + idpattern + r')/?$', 'enhanced_ui'),
#         url(r'^enhanced_image_(?P<size>full|display)/(?P<job_id>' + idpattern + r')/?$', 'enhanced_image', name='enhanced_image'),
#     )
#     

from astrometry.net.views.album import album, delete as album_delete, edit, new as album_new
urlpatterns.extend([
    re_path(r'^albums/(?P<album_id>' + idpattern + r')/?$', album, name='album'),
    re_path(r'^albums/(?P<album_id>' + idpattern + r')/delete/?$', album_delete, name='album_delete'),
    re_path(r'^albums/(?P<album_id>' + idpattern + r')/edit/?$', edit, name='album_edit'),
    re_path(r'^albums/new/?$', album_new, name='album_new'),
])

from astrometry.net.views.tag import index, delete, new, tag_autocomplete
urlpatterns.extend([
    re_path(r'^tags/?$', index, name='tags'),
    re_path(r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/tags/(?P<tag_id>' + tagpattern + r')/delete/?$', delete, name='tag_delete'),
    re_path(r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/tags/new/?$', new, name='tag_new'),
    re_path(r'^tags/autocomplete/?$', tag_autocomplete, name='tag_autocomplete'),
])

from astrometry.net.views.flag import update_flags
urlpatterns.append(
    re_path(r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/flags/update/?$', update_flags, name='update_flags'),
)

from astrometry.net.views.comment import new as new_comment, delete as delete_comment
urlpatterns.extend([
    re_path(r'^(?P<category>\w+)/(?P<recipient_id>' + idpattern + r')/comments/new/?$', new_comment, name='comment_new'),
    re_path(r'^comments/(?P<comment_id>' + idpattern + r')/delete/?$', delete_comment, name='comment_delete'),
])

from astrometry.net.views.license import edit
urlpatterns.append(
    re_path(r'^(?P<licensable_type>\w+)/(?P<licensable_id>' + idpattern + r')/license/edit/?$', edit, name='edit_license')
)


#     psidpattern = r'[0-9-]+'
#     
#     urlpatterns += patterns('astrometry.net.views.admin',
#                             (r'^admin/procsub/(?P<psid>'+psidpattern + r')?$', 'procsub'),
#                             (r'^admin/?', 'index'),
#                             )

from astrometry.net.api import (
    api_login, api_upload, url_upload, api_sdss_image_for_wcs, api_galex_image_for_wcs,
    api_submission_images, submission_status, myjobs, job_status, calibration, tags,
    machine_tags, objects_in_field, annotations_in_field, job_info, jobs_by_tag)

urlpatterns.extend([
    re_path(r'^api/login/?$', api_login, name='api_login'),
    re_path(r'^api/upload/?$', api_upload, name='api_upload'),
    re_path(r'^api/url_upload/?$', url_upload, name='api_url_upload'),
    re_path(r'^api/sdss_image_for_wcs/?$', api_sdss_image_for_wcs, name='api_sdss_image_for_wcs'),
    re_path(r'^api/galex_image_for_wcs/?$', api_galex_image_for_wcs, name='api_galex_image_for_wcs'),
    re_path(r'^api/submission_images/?$', api_submission_images, name='api_submission_images'),
    re_path(r'^api/submissions/(?P<sub_id>' + idpattern + r')/?$', submission_status, name='api_submission_status'),
    re_path(r'^api/myjobs/?', myjobs, name='api_myjobs'),
    re_path(r'^api/jobs/(?P<job_id>' + idpattern + r')/?$', job_status, name='api_job_status'),
    re_path(r'^api/jobs/(?P<job_id>' + idpattern + r')/calibration/?$', calibration, name='api_calibration'),
    re_path(r'^api/jobs/(?P<job_id>' + idpattern + r')/tags/?$', tags, name='api_tags'),
    re_path(r'^api/jobs/(?P<job_id>' + idpattern + r')/machine_tags/?$', machine_tags, name='api_machine_tags'),
    re_path(r'^api/jobs/(?P<job_id>' + idpattern + r')/objects_in_field/?$', objects_in_field, name='api_objects_in_field'),
    re_path(r'^api/jobs/(?P<job_id>(' + idpattern + r'|JOBID))/annotations/?$', annotations_in_field, name='api_annotations_in_field'),
    re_path(r'^api/jobs/(?P<job_id>' + idpattern + r')/info/?$', job_info, name='api_job_info'),
    re_path(r'^api/jobs_by_tag/?$', jobs_by_tag, name='api_jobs_by_tag'),
    #(r'^api/logout/?', 'logout'),
])

#     
#     # static file serving in development
#     if settings.DEBUG:
#         urlpatterns += patterns('',
#             (r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.STATICFILES_DIRS[0]}),
#         )

# fallback
from astrometry.net.views.home import home
urlpatterns.append(
    re_path(r'', home, name='home'),
)
