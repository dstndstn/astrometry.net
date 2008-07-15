from django.conf.urls.defaults import *

urlpatterns = (patterns('astrometry.net.portal.newjob',
                        (r'^newurl/$',          'newurl'  ),
                        (r'^newfile/$',         'newfile' ),
                        (r'^newlong/$',         'newlong' ),
                        ) +
               patterns('astrometry.net.portal.views',
                        (r'^status/$',          'jobstatus'),
                        (r'^getfile/$',         'getfile'  ),
                        (r'^joblist/$',         'joblist'  ),
                        (r'^summary/$',         'summary'  ),
                        (r'^set_description/$', 'job_set_description'),
                        (r'^substatusxml/$',    'submission_status_xml'),
                        (r'^changeperms/$',     'changeperms' ),
                        #(r'^publishtovo/$',    'publishtovo'),
                        # PLAY
                        (r'^redgreen$',     'redgreen'    ),
                        (r'^run-variant/$', 'run_variant' ),
                        ) +
               patterns('astrometry.net.portal.tags',
                        (r'^taglist/$',         'taglist'  ),
                        (r'^add_tag/$',         'job_add_tag' ),
                        (r'^remove_tag/$',      'job_remove_tag' ),
                        )
               )
