import unittest

from django.test.client import Client
from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User

from astrometry.net.portal.api import json2python, python2json
from astrometry.net.portal.job import Job, Tag

# Run with:
# cd ~/test/astrometry/net; python manage.py test portal.ApiTestCases

class ApiTestCases(TestCase):
    def setUp(self):
        super(ApiTestCases, self).setUp()
        self.loginurl = reverse('astrometry.net.portal.api.login')
        self.logouturl = reverse('astrometry.net.portal.api.logout')

        # create some dummy accounts - the Django test environment creates its own
        # temporary user database.
        self.u1 = 'test1@astrometry.net'
        self.p1 = 'password1'
        self.user1 = User.objects.create_user(self.u1, self.u1, self.p1)
        self.user1.save()

    def post_json(self, url, args):
        json = python2json(args)
        resp = self.client.post(url, { 'request-json': json })
        resp.json = json2python(resp.content)
        return resp

    def login_with(self, username, password):
        args = { 'username': username,
                 'password': password, }
        resp = self.post_json(self.loginurl, args)
        print 'response is', resp
        return resp

    def test_correct_login(self):
        resp = self.login_with(self.u1, self.p1)
        print 'response args: ', resp.json
        self.assert_('session' in resp.json)
        key = resp.json['session']
        args = {'session': key}
        r2 = self.post_json(reverse('astrometry.net.portal.api.amiloggedin'),
                            args)
        print 'response is', r2

    def test_logout(self):
        print
        print 'test_logout: logging in.'
        print
        resp = self.login_with(self.u1, self.p1)
        print 'response args: ', resp.json
        self.assert_('session' in resp.json)
        key = resp.json['session']
        args = {'session': key}
        print
        print 'test_logout: logging out.'
        print
        r2 = self.post_json(reverse('astrometry.net.portal.api.logout'),
                            args)
        print 'response is', r2
        print 'cookies:', self.client.cookies
        print 'session:', self.client.session
        
        #self.assert_(
            
    def test_job_status(self):
        j = Job(jobid='test-jobid-000', exposejob=True, status='Solved')
        j.save()
        t = Tag(job=j, machineTag=True, user=self.user1,
                text='robot_loves_this', addedtime=Job.timenow())
        t.save()
        t = Tag(job=j, machineTag=False, user=self.user1,
                text='nice shot', addedtime=Job.timenow())
        t.save()

        resp = self.login_with(self.u1, self.p1)
        key = resp.json['session']
        r2 = self.post_json(reverse('astrometry.net.portal.api.jobstatus'),
                            {'session': key,
                             'jobid': 'test-jobid-000',
                             })
        print 'response is', r2
        print 'cookies:', self.client.cookies
        print 'session:', self.client.session

