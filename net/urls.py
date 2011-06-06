from django.conf.urls.defaults import patterns, include, url

from astrometry.net import settings

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = (patterns('',
    (r'^login/$', include('django_openid_auth.urls')),
    (r'^logout/$', 'django.contrib.auth.views.logout'),
))


'''
			   patterns('astrometry.net1.portal.api',
						(r'^api/login', 'login'),
						(r'^api/logout', 'logout'),
						(r'^api/amiloggedin', 'amiloggedin'),
						(r'^api/jobstatus', 'jobstatus'),
						(r'^api/substatus', 'substatus'),
						(r'^api/submit_url', 'submit_url'),
						)
			   )
'''
						
