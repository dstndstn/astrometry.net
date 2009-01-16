from django.conf.urls.defaults import *
from django.contrib import admin

from astrometry.net import settings
#from astrometry.net.portal.models import *
import astrometry.net.portal.admin

admin.autodiscover()

urlpatterns = (patterns('',
                        (r'^tile/', include('astrometry.net.tile.urls')),
                        (r'^upload/', include('astrometry.net.upload.urls')),
                        (r'^job/', include('astrometry.net.portal.urls')),
                        (r'^admin/(.*)', admin.site.root),
                        )
               +
               patterns('django.contrib.auth.views',
                        (r'^login/', 'login',
                         {'template_name': 'portal/login.html'}),
                        (r'^changepassword/$',  'password_change',
                         {'template_name': 'portal/changepassword.html'}),
                        (r'^changepassword/done/', 'password_change_done',
                         {'template_name': 'portal/changedpassword.html'}),
                        (r'^resetpassword/',   'password_reset',
                         {'template_name': 'portal/resetpassword.html'}),
                        )
               +
               patterns('astrometry.net.portal.views',
                        (r'^logout/', 'logout'),
                        (r'^userprefs/', 'userprefs'),
                        (r'^newaccount/', 'newaccount'),
                        )
               +
               patterns('',
                        #(r'^vo/', include('astrometry.net.vo.urls')),
                        #(r'^testbed/', include('astrometry.net.testbed.urls')),
                        (r'^gmaps$', 'astrometry.net.tile.views.index'),
                        #(r'^hoggthinksimg', 'astrometry.net.portal.hoggthinks.image'),
                        #(r'^hoggthinks', 'astrometry.net.portal.hoggthinks.form'),
                        #(r'^easy-gmaps', 'astrometry.net.portal.easy_gmaps.tile'),
                        #

                        (r'^$', 'astrometry.net.portal.newjob.newurl'),
                        #(r'^$', 'astrometry.net.portal.newjob.newlong'),

                        # These are fake placeholders to allow {% url %} and reverse() to resolve an.media to /anmedia.
                        # -> They have corresponding fake definitions in astrometry/net/__init__.py
                        # -> You also have to set the Apache url match.
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
                        (r'^api/jobstatus', 'jobstatus'),
                        )
               )


