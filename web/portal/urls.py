from django.conf.urls.defaults import *

#from astrometry.web.portal import newjob
import astrometry.web.portal.newjob
import astrometry.web.portal.views

urlpatterns = patterns('',
					   (r'^newurl/$',      astrometry.web.portal.newjob.newurl),
					   (r'^newfile/$',     astrometry.web.portal.newjob.newfile),
					   (r'^newlong/$',     astrometry.web.portal.newjob.newlong),
					   (r'^status/$',      astrometry.web.portal.views.jobstatus),
					   (r'^getfile/$',     astrometry.web.portal.views.getfile),
					   (r'^joblist/$',     astrometry.web.portal.views.joblist),
					   (r'^taglist/$',     astrometry.web.portal.views.taglist),
					   (r'^summary/$',     astrometry.web.portal.views.summary),
					   (r'^set_description/$', astrometry.web.portal.views.job_set_description),
					   (r'^add_tag/$', astrometry.web.portal.views.job_add_tag),
					   (r'^remove_tag/$', astrometry.web.portal.views.job_remove_tag),
					   (r'^substatusxml/$',      astrometry.web.portal.views.submission_status_xml),
                       # PLAY
					   (r'^redgreen$',     astrometry.web.portal.views.redgreen),
                       (r'^run-variant/$', astrometry.web.portal.views.run_variant),
					   (r'^changeperms/$', astrometry.web.portal.views.changeperms),
					   #(r'^publishtovo/$', an.portal.views.publishtovo),
					   )
