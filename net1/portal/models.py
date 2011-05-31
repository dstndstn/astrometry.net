import logging
import os.path
import sha

from django.db import models
from django.contrib.auth.models import User
#from django.core import validators

import astrometry.net.settings as settings
from astrometry.net.portal.log import log

class UserProfile(models.Model):
	user = models.ForeignKey(User, unique=True, related_name='profile',
							 editable=False)

	activated = models.BooleanField(default=False)
	activation_key = models.CharField(max_length=20)

	# Automatically allow anonymous access to my job status pages?
	exposejobs = models.BooleanField(default=False)

	def __str__(self):
		s = ('UserProfile: user %s, activated: ' % self.user.username)
		if self.activated:
			s += 'yes'
		else:
			s += 'no, key=%s' % self.activation_key
		return s
	

	def new_activation_key(self):
		self.activation_key = User.objects.make_random_password(length=20)
		self.save()
		return self.activation_key

	def expose_jobs(self):
		return self.exposejobs

	def set_expose_jobs(self, tf):
		self.exposejobs = tf

	@staticmethod
	def for_user(user):
		if user is None:
			# HACK - default user profile.
			return UserProfile(user=None, exposejobs=True)
		if user.profile.count() == 0:
			# add new default profile.
			pro = UserProfile(user=user)
			pro.save()
		return user.get_profile()


from astrometry.net.portal.job import *
from astrometry.net.portal.wcs import *
from astrometry.net.portal.queue import *
