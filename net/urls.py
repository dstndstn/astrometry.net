from django.conf.urls.defaults import *
from astrometry.web import settings

urlpatterns = patterns('',
					   (r'^tile/', include('astrometry.web.tile.urls')),
					   (r'^upload/', include('astrometry.web.upload.urls')),
					   (r'^login/', 'django.contrib.auth.views.login',
                        {'template_name': 'portal/login.html'}),
                       (r'^logout/', 'django.contrib.auth.views.logout_then_login'),
					   (r'^userprefs/', 'astrometry.web.portal.views.userprefs'),
                       (r'^changepassword/$',  'django.contrib.auth.views.password_change',
                        {'template_name': 'portal/changepassword.html'}),
                       (r'^changepassword/done/', 'django.contrib.auth.views.password_change_done',
                        {'template_name': 'portal/changedpassword.html'}),
                       (r'^resetpassword/',   'django.contrib.auth.views.password_reset',
                        {'template_name': 'portal/resetpassword.html'}),
                       (r'^job/', include('astrometry.web.portal.urls')),
					   #(r'^vo/', include('astrometry.web.vo.urls')),
                       #(r'^testbed/', include('astrometry.web.testbed.urls')),
                       (r'^gmaps/$', 'astrometry.web.tile.views.index'),
                       #(r'^hoggthinksimg', 'astrometry.web.portal.hoggthinks.image'),
                       #(r'^hoggthinks', 'astrometry.web.portal.hoggthinks.form'),
                       #(r'^easy-gmaps', 'astrometry.web.portal.easy_gmaps.tile'),
                       #
                       (r'^$', 'astrometry.web.portal.newjob.newlong'),
                       # This is a fake placeholder to allow {% url %} and reverse() to resolve an.media to /anmedia.
                       # (see also media() in an/__init__.py)
                       (r'^anmedia/', 'astrometry.web.media'),
                       (r'^logout/', 'astrometry.web.logout'),
                       (r'^login/', 'astrometry.web.login'),
                       (r'^changepassword/',  'astrometry.web.changepassword'),
                       (r'^resetpassword/',   'astrometry.web.resetpassword'),
					   )


