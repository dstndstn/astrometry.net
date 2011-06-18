import random

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
