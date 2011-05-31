import tempfile
import os
import unittest
#from django.test.client import Client
from django.test import TestCase
#from django.core.urlresolvers import reverse
#from django.contrib.auth.models import User

from astrometry.net1.portal.job import DiskFile

class DiskFileTestCases(TestCase):
    #def setUp(self):
    #    super(DiskFileTestCases, self).setUp()

    def testForFile(self):
        (fd, tmpfile) = tempfile.mkstemp('', 'test-diskfile-1')
        os.close(fd)
        f = open(tmpfile, 'wb')
        f.write('Testing, Testing, 1, 2, 3.')
        f.close()
        df = DiskFile.for_file(tmpfile)
        df.save()

        # I got this by running "sha1sum" on the command-line on the string above.
        truehash = 'a7693cf965e3b92be36ff39b27b94637be30fc46'

        truepathend = 'a7/693cf965e3b92be36ff39b27b94637be30fc46'

        self.assertEqual(df.filehash, truehash)
        self.assert_(df.get_path().endswith(truepathend))
        self.assert_(os.path.exists(df.get_path()))

        # Create another file with the same contents and ensure that the
        # existing one is returned.
        (fd, tmpfile) = tempfile.mkstemp('', 'test-diskfile-2')
        os.close(fd)
        f = open(tmpfile, 'wb')
        f.write('Testing, Testing, 1, 2, 3.')
        f.close()
        df = DiskFile.for_file(tmpfile)

        # the existing one was returned - so the original file wasn't moved.
        self.assert_(os.path.exists(tmpfile))
        self.assert_(os.path.exists(df.get_path()))

        # clean up
        df.delete_file()


