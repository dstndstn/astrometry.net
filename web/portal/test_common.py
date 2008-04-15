import unittest

from django.test.client import Client
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.urlresolvers import reverse

from an.util.w3c_validator import W3CValidator

import an.gmaps_config as config

class PortalTestCase(TestCase):
    def setUp(self):
        super(PortalTestCase, self).setUp()
        self.urlprefix = 'http://testserver'
        # create some dummy accounts
        self.u1 = 'test1@astrometry.net'
        self.p1 = 'password1'
        accts = [ (self.u1, self.p1),
                  ('test2@astrometry.net', 'password2'), ]
        for (e, p) in accts:
            User.objects.create_user(e, e, p).save()
        self.loginurl = reverse('an.login')
        self.logouturl = reverse('an.logout')

    def login1(self):
        self.client.login(username=self.u1, password=self.p1)

    def validatePage(self, url):
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        v = W3CValidator(config.w3c_validator_url)
        self.assert_(v.validateText(resp.content))


