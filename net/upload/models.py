from django.db import models
from django.contrib.auth.models import User
#from django.core import validators
import sha
import re
import time
import random
import os.path
import logging

from astrometry.net import settings

logfile = settings.PORTAL_LOGFILE
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=logfile,
                    )


class UploadedFile(models.Model):
    uploadid_re = re.compile('[A-Za-z0-9]{%i}$' % (sha.digest_size*2))
    try:
        default_base_dir = os.environ['UPLOAD_DIR']
    except KeyError:
        default_base_dir = '/tmp'

    @staticmethod
    def isValidId(uid):
        logging.debug("IsValidId: " + str(uid))
        return UploadedFile.uploadid_re.match(uid)

    @staticmethod
    def generateId():
        h = sha.new()
        h.update(str(time.time()) + str(random.random()))
        uid = h.hexdigest()
        return uid

    @classmethod
    def set_default_base_dir(self,  basedir):
        self.default_base_dir = basedir

    # (this has to be defined before the "uploadid" member variable)
    #def validateId(self, field_data, all_data):
    #    if not self.isValidId(field_data):
    #        raise validators.ValidationError('Invalid upload ID')

    uploadid = models.CharField(max_length=100,
                                primary_key=True,
                                #validator_list=[validateId],
                                )
    user = models.ForeignKey(User, editable=False,
                             blank=True, null=True)

    userfilename = models.CharField(max_length=256, editable=False, blank=True, null=True)

    starttime = models.PositiveIntegerField(default=0)
    nowtime = models.PositiveIntegerField(default=0)
    predictedsize = models.PositiveIntegerField(default=0)
    byteswritten = models.PositiveIntegerField(default=0)
    filesize = models.PositiveIntegerField(default=0)
    errorstring = models.CharField(max_length=256, null=True, default='')

    def __init__(self, *args, **kwargs):
        logging.debug('UploadedFile.__init__()')
        for (x,y) in kwargs.items():
            logging.debug("kwarg[%s] = %s" % (str(x), str(y)))
        super(UploadedFile, self).__init__(*args, **kwargs)
        logging.debug('setting base dir.')
        if 'upload_base_dir' in kwargs:
            self.set_base_dir(kwargs['upload_base_dir'])
        else:
            self.set_base_dir(UploadedFile.default_base_dir)
        #logging.debug('UploadedFile (init): base dir is', getattr(self, 'upload_base_dir', None))
        #logging.debug("Set upload id to " + str(self.uploadid))
        #logging.debug("

    def __str__(self):
        return self.uploadid

    #def save(self):
    #    logging.debug("Save(): uploadid is " + str(self.uploadid))
    #    super(UploadedFile, self).save()

    #def set_upload_id(self, uid):
    #    self.uploadid = uid
    #    self.save()

    def set_base_dir(self, bdir):
        logging.debug('UploadedFile: setting base dir to', bdir)
        self.upload_base_dir = bdir

    def get_base_dir(self):
        return getattr(self, 'upload_base_dir', UploadedFile.default_base_dir)

    def xml(self):
        err = self.errorstring
        logging.debug('Error: %s' % err)
        args = {}
        dt = self.nowtime - self.starttime
        args['elapsed'] = dt
        if self.filesize > 0:
            sz = self.filesize
        else:
            sz = self.predictedsize
        args['filesize'] = sz
        if sz:
            args['pct'] = int(round(100 * self.byteswritten / float(sz)))
        if self.byteswritten and dt>0:
            avgspeed = self.byteswritten / float(dt)
            args['speed'] = avgspeed
            left = sz - self.byteswritten
            if left < 0:
                left = 0
            eta = left / avgspeed
            args['eta'] = eta
        if self.errorstring:
            args = {}
            args['error'] = self.errorstring

        tag = '<progress'
        for k,v in args.items():
            tag += ' ' + k + '="' + str(v) + '"'
        tag += ' />'
        return tag

    def get_filename(self):
        #logging.debug('UploadedFile.get_filename: base dir', getattr(self, 'upload_base_dir', None))
        fn = os.path.join(self.get_base_dir(), 'upload-' + self.uploadid)
        logging.debug('UploadedFile.get_filename: returning', fn)
        return fn

    def fileExists(self):
        path = self.get_filename()
        return os.path.exists(path)

        


