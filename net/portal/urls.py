from django.conf.urls.defaults import *

jobpattern = r'[a-z0-9-]+'

urlpatterns = (patterns('astrometry.net.portal.newjob',
                        (r'^newurl/$',          'newurl'  ),
                        (r'^newfile/$',         'newfile' ),
                        (r'^newlong/$',         'newlong' ),
                        ) +
               patterns('astrometry.net.portal.views',
                        (r'^status/(?P<jobid>' + jobpattern + r')', 'jobstatus'),
                        (r'^getfile/(?P<jobid>' + jobpattern + r')/(?P<filename>[a-z0-9.-]+)$', 'getfile'),
                        (r'^joblist/$',         'joblist'  ),
                        (r'^summary/$',         'summary'  ),
                        (r'^set_description/$', 'job_set_description'),
                        (r'^changeperms/$',     'changeperms' ),
                        #(r'^publishtovo/$',    'publishtovo'),
                        # PLAY
                        (r'^run-variant/$', 'run_variant' ),
                        ) +
               patterns('astrometry.net.portal.tags',
                        (r'^taglist/$',         'taglist'  ),
                        (r'^add_tag/$',         'job_add_tag' ),
                        (r'^remove_tag/$',      'job_remove_tag' ),
                        ) +
               patterns('astrometry.net.portal.redgreen',
                        (r'^redgreen$',     'redgreen'    ),
                        ) +
               patterns('astrometry.net.portal.legacy',
                        (r'^status/$',          'jobstatus_old'),
                        )
               )
