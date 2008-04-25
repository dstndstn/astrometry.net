import unittest

from django.test.client import Client
from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User

import django.newforms as forms

from astrometry.net.portal import views
from astrometry.net.portal.test_common import PortalTestCase

class LoginTestCases(PortalTestCase):
    def setUp(self):
        super(LoginTestCases, self).setUp()

    def login_with(self, username, password):
        resp = self.client.post(self.loginurl, { 'username': username, 'password': password })
        return resp

    def testLoginFormValid(self):
        print 'Validating login form:'
        self.validatePage(self.loginurl)

    def testLoginForm(self):
        print 'Login url is %s' % self.loginurl
        resp = self.client.get(self.loginurl)
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'portal/login.html')
        # print 'Got: %s' % resp.content

    def testEmptyUsername(self):
        resp = self.login_with('', 'pass')

        #print 'resp.context is', resp.context
        for c in resp.context:
            if not 'form' in c:
                continue
            #print 'form is in context', c
            f = c['form']
            print 'form type is', type(f)
            print 'form is', f

        #self.assertFormError(resp, 'form', 'username', 'This field is required.')
        self.assertOldFormError(resp, 'form', 'username', 'This field is required.')

    def testEmptyPassword(self):
        resp = self.login_with('bob', '')
        self.assertFormError(resp, 'form', 'password', 'This field is required.')

    def testEmptyBoth(self):
        resp = self.login_with('', '')
        self.assertFormError(resp, 'form', 'username', 'This field is required.')
        self.assertFormError(resp, 'form', 'password', 'This field is required.')

    def assertBadUsernamePasswordPair(self, resp):
        self.assertFormError(resp, 'form', None,#'__all__',
                             'Please enter a correct username and password. Note that both fields are case-sensitive.')

    def testBadUsername(self):
        resp = self.login_with('u', 'smeg')
        self.assertBadUsernamePasswordPair(resp)

    def testBadPassword(self):
        resp = self.login_with(self.u1, 'smeg')
        self.assertBadUsernamePasswordPair(resp)
        #for c in resp.context:
        #    errs = c['form'].errors
        #    for f,e in errs.items():
        #        print 'Field "%s": error "%s"' % (f, e)

    def testNoSuchUser(self):
        resp = self.login_with('nobody@nowhere.com', 'smeg')
        self.assertBadUsernamePasswordPair(resp)

    # FIXME - this should maybe move to test_job_summary:
    def testJobSummaryRedirects(self):
        url = reverse('astrometry.net.portal.views.summary')
        resp = self.client.get(url)
        redirurl = self.urlprefix + self.loginurl + '?next=' + url
        self.assertRedirects(resp, redirurl)

    # FIXME - add other redirect tests.

    # check that when a user is logged in, they don't get redirected to the
    # login page.
    def testLoggedInNoRedirect(self):
        url = reverse('astrometry.net.portal.views.summary')
        self.login1()
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)

    def testLoginWorks(self):
        resp = self.login_with(self.u1, self.p1)
        url = reverse('astrometry.net.portal.newjob.newurl')
        self.assertRedirects(resp, self.urlprefix + url)

    def testLogoutRedirectsToLogin(self):
        resp = self.login_with(self.u1, self.p1)
        url = reverse('astrometry.net.portal.newjob.newurl')
        # logged in.
        self.assertRedirects(resp, self.urlprefix + url)
        resp = self.client.post(self.logouturl)
        # logged out.
        self.assertRedirects(resp, self.urlprefix + self.loginurl)
        # after logout, attempts to access regular pages should
        # redirect to the login page.
        self.testJobSummaryRedirects()
