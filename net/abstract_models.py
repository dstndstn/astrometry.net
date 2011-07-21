from log import *

import urllib
import urllib2

import xml.dom.minidom
from xml.dom.minidom import Node

from django.db import models
from django.contrib.auth.models import User
from astrometry.net.settings import *

class Commentable(models.Model):
    id = models.AutoField(primary_key=True)
    owner = models.ForeignKey(User, null=True)


class Hideable(models.Model):
    YES_NO = (('y','yes'),('n','no'))

    publicly_visible = models.CharField(choices=YES_NO, max_length=1,
        default='y')

    share_with = models.ManyToManyField(User, through="SharedHideable")
    
    def share_with(self, user):
        shared_hideable = SharedHideable(hideable=self, shared_with=user)
        shared_hideable.save()
        return shared_hideable;

    def unhide(self):
        self.publicly_visible = 'y'
        self.save()

    def hide(self):
        self.publicly_visible = 'n'
        self.save()

    def is_public(self):
        return self.publicly_visible == 'y'


class SharedHideable(models.Model):
    hideable = models.ForeignKey('Hideable')
    shared_with = models.ForeignKey(User)
    
    # so it may be possible in the future to include "recently shared with
    # you" view
    created_at = models.DateTimeField(auto_now_add=True)
    

# uses creative commons rest api
class Licensable(models.Model):
    class Meta:
        abstract = True

    YES_NO = (
        ('y','yes'),
        ('n','no'),
        ('d','use default'),
    )
    YES_SA_NO = (
        ('y','yes'),
        ('sa','yes, but share alike'),
        ('n','no'),
        ('d','use default'),
    )

    # CC "answer" data
    allow_commercial_use = models.CharField(choices=YES_NO, max_length=1,
        default='d')
    allow_modifications = models.CharField(choices=YES_SA_NO, max_length=2,
        default='d')

    # CC issued license
    license_name = models.CharField(max_length=1024)
    license_uri = models.CharField(max_length=1024)

    # attribution and other optional fields here


    def get_license_uri(self):
        if self.license_uri == '':
            self.save()
        return self.license_uri

    def get_license_name(self):
        if self.license_name == '':
            self.save()
        return self.license_name

    # replaces 'd' with actual setting from default license
    def replace_license_default(self, default):
        if self.allow_commercial_use == 'd':
            self.allow_commercial_use = default.allow_commercial_use
        if self.allow_modifications == 'd':
            self.allow_modifications = default.allow_modifications

    def get_license_xml(self):
        try:
            allow_commercial_use = self.allow_commercial_use
            allow_modifications = self.allow_modifications

            url = (
                'http://api.creativecommons.org/rest/1.5/license/standard/get?commercial=%s&derivatives=%s&jurisdiction=' %
                (allow_commercial_use,
                allow_modifications,)
            )
            logmsg("getting license via url: %s" % url)
            f = urllib2.urlopen(url)
            xml = f.read()
            f.close()
            return xml
        except Exception as e:
            logmsg('error getting license xml: %s' % str(e))
            return None

    # uses CC "answer" data to retrieve CC issued license data
    def get_license_name_uri(self):
        def get_text(node_list):
            rc = []
            for node in node_list:
                if node.nodeType == node.TEXT_NODE:
                    rc.append(node.data)
            return ''.join(rc)
        try:
            license_xml = self.get_license_xml()
            license_doc = xml.dom.minidom.parseString(license_xml)
            self.license_name = get_text(license_doc.getElementsByTagName('license-name')[0].childNodes)
            self.license_uri = get_text(license_doc.getElementsByTagName('license-uri')[0].childNodes)
            # can add rdf stuff here if we want..
            
        except Exception as e:
            logmsg('error getting issued license data: %s' % str(e))
