import unittest

from django.test.client import Client
from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User

from astrometry.net.portal.api import json2python, python2json

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
        accts = [ (self.u1, self.p1), ]
        for (e, p) in accts:
            User.objects.create_user(e, e, p).save()

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
            
