import unittest

from django.test.client import Client
from django.test import TestCase
from django.contrib.auth.models import User
from django.core.urlresolvers import reverse

from astrometry.util.w3c_validator import W3CValidator
import astrometry.net.settings as settings

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
        self.loginurl = reverse('astrometry.net.login')
        self.logouturl = reverse('astrometry.net.logout')

    def login1(self):
        self.client.login(username=self.u1, password=self.p1)

    def validatePage(self, url):
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        print 'W3C Validation url: ', settings.W3C_VALIDATOR_URL
        v = W3CValidator(settings.W3C_VALIDATOR_URL)
        ok = v.validateText(resp.content)
        if ok:
            print 'Validation passed.'
        else:
            print 'Validation failed.'
        self.assert_(ok)


