from unittest import TestLoader, TestSuite

from astrometry.net.portal.test_login import LoginTestCases
from astrometry.net.portal.test_newjob import NewJobTestCases
from astrometry.net.portal.test_diskfile import DiskFileTestCases
from astrometry.net.portal.test_api import ApiTestCases
from astrometry.net.portal.test_accounts import AccountTestCases

def suite():
    all_suites = TestSuite()

    for x in [ AccountTestCases, LoginTestCases, NewJobTestCases, DiskFileTestCases, ApiTestCases ]:
        suite  = TestLoader().loadTestsFromTestCase(x)
        all_suites.addTest(suite)

    return all_suites
