from django.conf.urls.defaults import *

urlpatterns = patterns('astrometry.net.portal',
					   (r'^newurl/$',          'newjob.newurl'  ),
					   (r'^newfile/$',         'newjob.newfile' ),
					   (r'^newlong/$',         'newjob.newlong' ),
					   (r'^status/$',          'views.jobstatus'),
					   (r'^getfile/$',         'views.getfile'  ),
					   (r'^joblist/$',         'views.joblist'  ),
					   (r'^summary/$',         'views.summary'  ),
					   (r'^set_description/$', 'views.job_set_description'),
					   (r'^taglist/$',         'tags.taglist'  ),
					   (r'^add_tag/$',         'tags.job_add_tag' ),
					   (r'^remove_tag/$',      'tags.job_remove_tag' ),
					   (r'^substatusxml/$',    'views.submission_status_xml'),
					   (r'^changeperms/$',     'views.changeperms' ),
					   #(r'^publishtovo/$',    'views.publishtovo'),
                       # PLAY
					   (r'^redgreen$',     'views.redgreen'    ),
                       (r'^run-variant/$', 'views.run_variant' ),
					   )
