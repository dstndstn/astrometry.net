import unittest
from django.test.client import Client
from django.test import TestCase

from django.core.urlresolvers import reverse

from django.contrib.auth.models import User
from astrometry.net.portal import views

from astrometry.net.portal.test_common import PortalTestCase

class NewJobTestCases(PortalTestCase):
    def setUp(self):
        super(NewJobTestCases, self).setUp()
        self.joblongurl = reverse('astrometry.net.portal.newjob.newlong')
        self.joburlurl  = reverse('astrometry.net.portal.newjob.newurl')
        self.jobfileurl = reverse('astrometry.net.portal.newjob.newfile')

    def assertRedirectsToLogin(self, url):
        # When not logged in, it should redirect to the login page.
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 302)
        redir = self.urlprefix + self.loginurl + '?next=' + url
        self.assertEqual(resp['Location'], redir)

    def testLongFormRedirects(self):
        self.assertRedirectsToLogin(self.joblongurl)
    def testURLFormRedirects(self):
        self.assertRedirectsToLogin(self.joburlurl)
    def testFileFormRedirects(self):
        self.assertRedirectsToLogin(self.jobfileurl)

    def assertNoRedirect(self, url, template):
        self.login1()
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, template)

    def testLongFormNoRedirect(self):
        self.assertNoRedirect(self.joblongurl, 'portal/newjoblong.html')
    def testURLFormNoRedirect(self):
        self.assertNoRedirect(self.joburlurl,  'portal/newjoburl.html' )
    def testFileFormNoRedirect(self):
        self.assertNoRedirect(self.jobfileurl, 'portal/newjobfile.html')

    def testLongFormValid(self):
        self.login1()
        print 'Validating long job form:'
        self.validatePage(self.joblongurl)

    def testURLFormValid(self):
        self.login1()
        print 'Validating URL job form:'
        self.validatePage(self.joburlurl)

    def testFileFormValid(self):
        self.login1()
        print 'Validating file job form:'
        self.validatePage(self.jobfileurl)

