#! /usr/bin/env python

import os
import sys
import tempfile
import traceback

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
sys.path.extend(['/home/gmaps/test',
                 '/home/gmaps/django/lib/python2.4/site-packages'])

import astrometry.net.settings as settings

os.environ['LD_LIBRARY_PATH'] = settings.UTIL_DIR
os.environ['PATH'] = ':'.join(['/bin', '/usr/bin',
                               settings.BLIND_DIR,
                               settings.UTIL_DIR,
                               ])
# This must match the Apache setting UPLOAD_DIR
os.environ['UPLOAD_DIR'] = settings.UPLOAD_DIR

import logging
import os.path
import urllib
import urllib2
import shutil
import tarfile

from urlparse import urlparse
from urllib import urlencode
from urllib2 import urlopen
from StringIO import StringIO

from django.db import models

from astrometry.net.portal.models import Job, Submission, DiskFile, Calibration, Tag
from astrometry.net.upload.models import UploadedFile
from astrometry.net.portal.log import log
from astrometry.net.portal.convert import convert, is_tarball, FileConversionError
from astrometry.net.portal.wcs import TanWCS
from astrometry.util.run_command import run_command
from astrometry.util.file import *

from astrometry.net.server import ssh_master

import astrometry.util.sip as sip

class WatcherDsolver(Watcher):
    def solve_job(self, job):
        def logfunc(s):
            f = open(job.get_filename('blind.log'), 'a')
            f.write(s)
            f.close()
        tardata = ssh_master.solve(job, logfunc)
        return tardata


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s <input-file>' % sys.argv[0]
        sys.exit(-1)
    joblink = sys.argv[1]
    w = WatcherDsolver()
    sys.exit(w.main(joblink))

