from django.conf.urls.defaults import *
from astrometry.net import settings

urlpatterns = (patterns('',
                        (r'^tile/', include('astrometry.net.tile.urls')),
                        #(r'^shard', include('astrometry.net.server.shardurls')),
                        #(r'^master', include('astrometry.net.server.masterurls')),
                        (r'^upload/', include('astrometry.net.upload.urls')),
                        (r'^login/', 'django.contrib.auth.views.login',
                         {'template_name': 'portal/login.html'}),
                        (r'^logout/', 'django.contrib.auth.views.logout_then_login'),
                        (r'^userprefs/', 'astrometry.net.portal.views.userprefs'),
                        (r'^changepassword/$',  'django.contrib.auth.views.password_change',
                         {'template_name': 'portal/changepassword.html'}),
                        (r'^changepassword/done/', 'django.contrib.auth.views.password_change_done',
                         {'template_name': 'portal/changedpassword.html'}),
                        (r'^resetpassword/',   'django.contrib.auth.views.password_reset',
                         {'template_name': 'portal/resetpassword.html'}),
                        (r'^job/', include('astrometry.net.portal.urls')),
                        #(r'^vo/', include('astrometry.net.vo.urls')),
                        #(r'^testbed/', include('astrometry.net.testbed.urls')),
                        (r'^gmaps$', 'astrometry.net.tile.views.index'),
                        #(r'^hoggthinksimg', 'astrometry.net.portal.hoggthinks.image'),
                        #(r'^hoggthinks', 'astrometry.net.portal.hoggthinks.form'),
                        #(r'^easy-gmaps', 'astrometry.net.portal.easy_gmaps.tile'),
                        #
                        (r'^$', 'astrometry.net.portal.newjob.newlong'),
                        # These are fake placeholders to allow {% url %} and reverse() to resolve an.media to /anmedia.
                        # They have corresponding fake definitions in astrometry/net/__init__.py
                        (r'^anmedia/', 'astrometry.net.media'),
                        (r'^logout/', 'astrometry.net.logout'),
                        (r'^login/', 'astrometry.net.login'),
                        (r'^changepassword/',  'astrometry.net.changepassword'),
                        (r'^resetpassword/',   'astrometry.net.resetpassword'),
                        (r'^newaccount/',   'astrometry.net.newaccount'),
                        )
               +
               patterns('astrometry.net.portal.api',
                        (r'^api/login', 'login'),
                        (r'^api/logout', 'logout'),
                        (r'^api/amiloggedin', 'amiloggedin'),
                        )
               )


