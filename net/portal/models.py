import logging
import os.path
import sha

from django.db import models
from django.contrib.auth.models import User
#from django.core import validators

import astrometry.net.settings as settings
from astrometry.net.portal.log import log

class UserPreferences(models.Model):
    user = models.ForeignKey(User, editable=False)

    # Automatically allow anonymous access to my job status pages?
    exposejobs = models.BooleanField(default=False)

    def __str__(self):
        s = ('<UserPreferences: ' + self.user.username +
             ', expose jobs: ' + (self.exposejobs and 'T' or 'F'))
        return s

    @staticmethod
    def for_user(user):
        prefset = UserPreferences.objects.all().filter(user = user)
        if not prefset or not len(prefset):
            # no existing user prefs.
            prefs = UserPreferences(user = user)
        else:
            prefs = prefset[0]
        return prefs

    def expose_jobs(self):
        return self.exposejobs

    def set_expose_jobs(self, tf):
        self.exposejobs = tf


from astrometry.net.portal.job import *
from astrometry.net.portal.wcs import *
from astrometry.net.portal.queue import *
