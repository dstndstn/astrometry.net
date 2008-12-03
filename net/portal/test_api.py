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

    def login_with(self, username, password):
        args = { 'username': username,
                 'password': password, }
        json = python2json(args)
        resp = self.client.post(self.loginurl, { 'request-json': json })
        print 'response is', resp
        return resp

    def test_correct_login(self):
        resp = self.login_with(self.u1, self.p1)
        py = json2python(resp.content)
        print 'response args: ', py
        self.assert_('session' in py)
        key = py['session']
        args = {'session': key}
        json = python2json(args)
        r2 = self.client.post(reverse('astrometry.net.portal.api.amiloggedin'),
                              { 'request-json': json})
        print 'response is', r2

