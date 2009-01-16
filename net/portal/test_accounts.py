import unittest

from django.test.client import Client
from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User

from astrometry.net.portal import accounts
from astrometry.net.portal.test_common import PortalTestCase

class AccountTestCases(PortalTestCase):

    def testFormValid(self):
        print 'Validating newaccount form:'
        self.validatePage(reverse('astrometry.net.newaccount'))
