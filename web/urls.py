from django.conf.urls.defaults import *
from an import settings

urlpatterns = patterns('',
					   (r'^tile/', include('an.tile.urls')),
					   (r'^upload/', include('an.upload.urls')),
					   (r'^login/', 'django.contrib.auth.views.login',
                        {'template_name': 'portal/login.html'}),
                       (r'^logout/', 'django.contrib.auth.views.logout_then_login'),
					   (r'^userprefs/', 'an.portal.views.userprefs'),
                       (r'^changepassword/$',  'django.contrib.auth.views.password_change',
                        {'template_name': 'portal/changepassword.html'}),
                       (r'^changepassword/done/', 'django.contrib.auth.views.password_change_done',
                        {'template_name': 'portal/changedpassword.html'}),
                       (r'^resetpassword/',   'django.contrib.auth.views.password_reset',
                        {'template_name': 'portal/resetpassword.html'}),
                       (r'^job/', include('an.portal.urls')),
					   #(r'^vo/', include('an.vo.urls')),
                       (r'^testbed/', include('an.testbed.urls')),
                       (r'^gmaps/$', 'an.tile.views.index'),
                       (r'^hoggthinksimg', 'an.portal.hoggthinks.image'),
                       (r'^hoggthinks', 'an.portal.hoggthinks.form'),
                       (r'^easy-gmaps', 'an.portal.easy_gmaps.tile'),
                       #
                       (r'^$', 'an.portal.newjob.newlong'),
                       # This is a fake placeholder to allow {% url %} and reverse() to resolve an.media to /anmedia.
                       # (see also media() in an/__init__.py)
                       (r'^anmedia/', 'an.media'),
                       (r'^logout/', 'an.logout'),
                       (r'^login/', 'an.login'),
                       (r'^changepassword/',  'an.changepassword'),
                       (r'^resetpassword/',   'an.resetpassword'),
					   )


