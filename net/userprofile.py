import random

from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
	user = models.ForeignKey(User, unique=True, related_name='profile',
							 editable=False)

	apikey = models.CharField(max_length = 1024)
	
	def __str__(self):
		s = ('UserProfile: user %s, API key %s' % (self.user.get_full_name(), self.apikey))
		return s


	def create_api_key(self):
		key = ''.join([chr(random.randint(ord('a'), ord('z')))
					   for i in range(16)])
		self.apikey = key


