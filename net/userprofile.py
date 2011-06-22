import random

from django.core.urlresolvers import reverse
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    API_KEY_LENGTH = 16
    display_name = models.CharField(max_length=256)
    user = models.ForeignKey(User, unique=True, related_name='profile',
                             editable=False)
    apikey = models.CharField(max_length = API_KEY_LENGTH)
    
    def __str__(self):
        s = ('UserProfile: user %s, API key %s' % (self.user.get_full_name(), self.apikey))
        return s

    def create_api_key(self):
        key = ''.join([chr(random.randint(ord('a'), ord('z')))
                       for i in range(self.__class__.API_KEY_LENGTH)])
        self.apikey = key

    def get_absolute_url(self):
        return reverse('astrometry.net.views.user.public_profile', user_id=self.user.id)

    def save(self, *args, **kwargs):
        # for sorting users, enforce capitalization of first letter
        self.display_name = self.display_name[:1].capitalize() + self.display_name[1:]
        return super(UserProfile, self).save(*args, **kwargs)
