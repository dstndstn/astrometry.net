from log import *

import urllib
import urllib2

import xml.dom.minidom
from xml.dom.minidom import Node

from django.db import models
from django.contrib.auth.models import User
from astrometry.net.settings import *

class Hideable(models.Model):
    class Meta:
        abstract = True

    YES_NO = (('y','yes'),('n','no'))

    publicly_visible = models.CharField(choices=YES_NO, max_length=1,
        default='y')

    def unhide(self):
        self.publicly_visible = 'y'
        self.save()

    def hide(self):
        self.publicly_visible = 'n'
        self.save()

    def is_public(self):
        return self.publicly_visible == 'y'
