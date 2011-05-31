from django.conf.urls.defaults import *
from django.contrib import admin

from astrometry.net1 import settings
#from astrometry.net1.portal.models import *
import astrometry.net1.portal.admin

admin.autodiscover()

urlpatterns = (patterns('',
						(r'^tile/', include('astrometry.net1.tile.urls')),
						#(r'^upload/', include('astrometry.net1.upload.urls')),
						(r'^job/', include('astrometry.net1.portal.urls')),
						#(r'^admin/(.*)', admin.site.root),
						)
			   +
			   patterns('astrometry.net1.upload.views',
						(r'^upload/form/', 'uploadform',
						 {'template_name': 'portal/uploadfile.html',
						  'onload': 'parent.uploadframeloaded()',
						  'target': settings.UPLOADER_URL + '?onload=parent.uploadFinished()',
						  }),
						(r'^upload/formsmall/', 'uploadformsmall',
						 {'template_name': 'portal/uploadfilesmall.html',
						  'onload': 'parent.uploadframeloaded()',
						  'target': settings.UPLOADER_URL + '?onload=parent.uploadFinished()',
						  }),
						(r'^upload/progress_ajax/$', 'progress_ajax'),
						(r'^upload/xml/$',           'progress_xml' ),
						)
			   +
			   patterns('django.contrib.auth.views',
						(r'^login/', 'login',
						 {'template_name': 'portal/login.html'}),

						(r'^changepassword/$',	'password_change',
						 {'template_name': 'portal/changepassword.html'}),

						(r'^changepassword/done/', 'password_change_done',
						 {'template_name': 'portal/changedpassword.html'}),

						(r'^resetpassword/',   'password_reset',
						 {'template_name': 'portal/resetpassword.html',
						  'email_template_name': 'portal/resetpasswordemail.html'}),

						(r'^resetpassworddone/', 'password_reset_done',
						 {'template_name': 'portal/resetpassworddone.html'}),

						#(r'^resetpasswordconfirm/(?P<uidb36>[0-9A-Za-z]+)-(?P<token>.+)/$',
						# 'password_reset_confirm',
						# {'template_name': 'portal/resetpasswordconfirm.html'}),
						# To login session try:
						#   post_reset_redirect =
						#   set_password_form = <subclass of django.contrib.auth.forms.SetPasswordForm>
						#     override save()

						(r'^resetpasswordcomplete/', 'password_reset_complete'),
						)
			   +
			   patterns('astrometry.net1.portal.passwordreset',
						(r'^resetpasswordconfirm/(?P<uidb36>[0-9A-Za-z]+)-(?P<token>.+)/$',
						 'password_reset_confirm',
						 {'template_name': 'portal/resetpasswordconfirm.html',
						  'post_reset_redirect': settings.LOGIN_REDIRECT_URL}),
						)
			   +
			   patterns('astrometry.net1.portal.accounts',
						(r'^logout/', 'logout'),
						(r'^userprefs/', 'userprefs'),
						(r'^newaccount/activate', 'activateaccount'),
						(r'^newaccount/', 'newaccount'),
						(r'^setpassword/', 'setpassword'),
						)
			   +
			   patterns('',
						#(r'^vo/', include('astrometry.net1.vo.urls')),
						#(r'^testbed/', include('astrometry.net1.testbed.urls')),
						(r'^gmaps$', 'astrometry.net1.tile.views.index'),
						#(r'^hoggthinksimg', 'astrometry.net1.portal.hoggthinks.image'),
						#(r'^hoggthinks', 'astrometry.net1.portal.hoggthinks.form'),
						#(r'^easy-gmaps', 'astrometry.net1.portal.easy_gmaps.tile'),
						#

						(r'^$', 'astrometry.net1.portal.newjob.newurl'),
						#(r'^$', 'astrometry.net1.portal.newjob.newlong'),

						# These are fake placeholders to allow {% url %} and reverse() to resolve an.media to /anmedia.
						# -> They have corresponding fake definitions in astrometry/net/__init__.py
						# -> You also have to set the Apache url match.
						(r'^media/(?P<filename>[\w.]+)$', 'astrometry.net1.portal.views.media'),
						(r'^anmedia/(?P<filename>[\w.]+)$', 'astrometry.net1.portal.views.media'),
						# ??
						(r'^anmedia/', 'astrometry.net1.media'),
						(r'^logout/', 'astrometry.net1.logout'),
						(r'^login/', 'astrometry.net1.login'),
						(r'^changepassword/',  'astrometry.net1.changepassword'),
						(r'^resetpassword/',   'astrometry.net1.resetpassword'),
						#(r'^resetpasswordconfirm/',   'astrometry.net1.resetpassword'),
						#(r'^resetpassword/',	'astrometry.net1.resetpassword'),
						#(r'^resetpassword/',	'astrometry.net1.resetpassword'),
						(r'^setpassword/',	 'astrometry.net1.setpassword'),
						(r'^newaccount/',	'astrometry.net1.newaccount'),
						)
			   +
			   patterns('astrometry.net1.portal.api',
						(r'^api/login', 'login'),
						(r'^api/logout', 'logout'),
						(r'^api/amiloggedin', 'amiloggedin'),
						(r'^api/jobstatus', 'jobstatus'),
						(r'^api/substatus', 'substatus'),
						(r'^api/submit_url', 'submit_url'),
						)
			   )


