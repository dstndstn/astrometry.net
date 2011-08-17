# encoding: utf-8
import datetime
from south.db import db
from south.v2 import DataMigration
from django.db import models

from astrometry.net.settings import *
from astrometry.util.util import Tan
import math
import os

class Migration(DataMigration):

    def forwards(self, orm):
        "Write your forwards methods here."
        
        for calib in orm.Calibration.objects.all():
            wcsfn = os.path.join(JOBDIR, '%08i' % calib.job.id)
            wcsfn = os.path.join(wcsfn, 'wcs.fits')
            wcs = Tan(str(wcsfn), 0)
            ra,dec = wcs.radec_center()
            radius = (wcs.pixel_scale() *
                math.hypot(wcs.imagew, wcs.imageh)/2. / 3600.)
            # Find cartesian coordinates
            ra *= math.pi/180
            dec *= math.pi/180
            tempr = math.cos(dec)
            calib.x = tempr*math.cos(ra)
            calib.y = tempr*math.sin(ra)
            calib.z = math.sin(dec)
            calib.r = radius/180*math.pi
            calib.save()


    def backwards(self, orm):
        "Write your backwards methods here."


    models = {
        'auth.group': {
            'Meta': {'object_name': 'Group'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '80'}),
            'permissions': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Permission']", 'symmetrical': 'False', 'blank': 'True'})
        },
        'auth.permission': {
            'Meta': {'ordering': "('content_type__app_label', 'content_type__model', 'codename')", 'unique_together': "(('content_type', 'codename'),)", 'object_name': 'Permission'},
            'codename': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'content_type': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['contenttypes.ContentType']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '50'})
        },
        'auth.user': {
            'Meta': {'object_name': 'User'},
            'date_joined': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime.now'}),
            'email': ('django.db.models.fields.EmailField', [], {'max_length': '75', 'blank': 'True'}),
            'first_name': ('django.db.models.fields.CharField', [], {'max_length': '30', 'blank': 'True'}),
            'groups': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Group']", 'symmetrical': 'False', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'is_active': ('django.db.models.fields.BooleanField', [], {'default': 'True'}),
            'is_staff': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'is_superuser': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'last_login': ('django.db.models.fields.DateTimeField', [], {'default': 'datetime.datetime.now'}),
            'last_name': ('django.db.models.fields.CharField', [], {'max_length': '30', 'blank': 'True'}),
            'password': ('django.db.models.fields.CharField', [], {'max_length': '128'}),
            'user_permissions': ('django.db.models.fields.related.ManyToManyField', [], {'to': "orm['auth.Permission']", 'symmetrical': 'False', 'blank': 'True'}),
            'username': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '30'})
        },
        'contenttypes.contenttype': {
            'Meta': {'ordering': "('name',)", 'unique_together': "(('app_label', 'model'),)", 'object_name': 'ContentType', 'db_table': "'django_content_type'"},
            'app_label': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'model': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'})
        },
        'net.album': {
            'Meta': {'object_name': 'Album'},
            'comment_receiver': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.CommentReceiver']", 'unique': 'True'}),
            'created_at': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'publicly_visible': ('django.db.models.fields.CharField', [], {'default': "'y'", 'max_length': '1'}),
            'tags': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'albums'", 'symmetrical': 'False', 'to': "orm['net.Tag']"}),
            'title': ('django.db.models.fields.CharField', [], {'max_length': '64'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'albums'", 'null': 'True', 'to': "orm['auth.User']"}),
            'user_images': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'albums'", 'symmetrical': 'False', 'to': "orm['net.UserImage']"})
        },
        'net.cachedfile': {
            'Meta': {'object_name': 'CachedFile'},
            'disk_file': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.DiskFile']"}),
            'key': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '64', 'primary_key': 'True'})
        },
        'net.calibration': {
            'Meta': {'object_name': 'Calibration'},
            'decmax': ('django.db.models.fields.FloatField', [], {}),
            'decmin': ('django.db.models.fields.FloatField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'r': ('django.db.models.fields.FloatField', [], {}),
            'ramax': ('django.db.models.fields.FloatField', [], {}),
            'ramin': ('django.db.models.fields.FloatField', [], {}),
            'raw_tan': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'calibrations_raw'", 'null': 'True', 'to': "orm['net.TanWCS']"}),
            'sip': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.SipWCS']", 'null': 'True'}),
            'sky_location': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'calibrations'", 'null': 'True', 'to': "orm['net.SkyLocation']"}),
            'tweaked_tan': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'calibrations_tweaked'", 'null': 'True', 'to': "orm['net.TanWCS']"}),
            'x': ('django.db.models.fields.FloatField', [], {}),
            'y': ('django.db.models.fields.FloatField', [], {}),
            'z': ('django.db.models.fields.FloatField', [], {})
        },
        'net.comment': {
            'Meta': {'ordering': "['-created_at']", 'object_name': 'Comment'},
            'author': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'comments_left'", 'to': "orm['auth.User']"}),
            'created_at': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'recipient': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'comments'", 'to': "orm['net.CommentReceiver']"}),
            'text': ('django.db.models.fields.CharField', [], {'max_length': '1024'})
        },
        'net.commentreceiver': {
            'Meta': {'object_name': 'CommentReceiver'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'owner': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['auth.User']", 'null': 'True'})
        },
        'net.diskfile': {
            'Meta': {'object_name': 'DiskFile'},
            'file_hash': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '40', 'primary_key': 'True'}),
            'file_type': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True'}),
            'size': ('django.db.models.fields.PositiveIntegerField', [], {})
        },
        'net.flag': {
            'Meta': {'ordering': "['name']", 'object_name': 'Flag'},
            'explanation': ('django.db.models.fields.CharField', [], {'max_length': '2048', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '56', 'primary_key': 'True'})
        },
        'net.flaggeduserimage': {
            'Meta': {'object_name': 'FlaggedUserImage'},
            'flag': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Flag']"}),
            'flagged_time': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['auth.User']"}),
            'user_image': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.UserImage']"})
        },
        'net.image': {
            'Meta': {'object_name': 'Image'},
            'disk_file': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.DiskFile']"}),
            'display_image': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'image_display_set'", 'null': 'True', 'to': "orm['net.Image']"}),
            'height': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'thumbnail': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'image_thumbnail_set'", 'null': 'True', 'to': "orm['net.Image']"}),
            'width': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True'})
        },
        'net.job': {
            'Meta': {'object_name': 'Job'},
            'calibration': ('django.db.models.fields.related.OneToOneField', [], {'related_name': "'job'", 'unique': 'True', 'null': 'True', 'to': "orm['net.Calibration']"}),
            'end_time': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'error_message': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'queued_time': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'start_time': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'status': ('django.db.models.fields.CharField', [], {'max_length': '1'}),
            'user_image': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'jobs'", 'to': "orm['net.UserImage']"})
        },
        'net.license': {
            'Meta': {'object_name': 'License'},
            'allow_commercial_use': ('django.db.models.fields.CharField', [], {'default': "'d'", 'max_length': '1'}),
            'allow_modifications': ('django.db.models.fields.CharField', [], {'default': "'d'", 'max_length': '2'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'license_name': ('django.db.models.fields.CharField', [], {'max_length': '1024'}),
            'license_uri': ('django.db.models.fields.CharField', [], {'max_length': '1024'})
        },
        'net.processsubmissions': {
            'Meta': {'object_name': 'ProcessSubmissions'},
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'pid': ('django.db.models.fields.IntegerField', [], {}),
            'watchdog': ('django.db.models.fields.DateTimeField', [], {'null': 'True'})
        },
        'net.queuedjob': {
            'Meta': {'object_name': 'QueuedJob'},
            'finished': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'job': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Job']"}),
            'procsub': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'jobs'", 'to': "orm['net.ProcessSubmissions']"}),
            'success': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'net.queuedsubmission': {
            'Meta': {'object_name': 'QueuedSubmission'},
            'finished': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'procsub': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'subs'", 'to': "orm['net.ProcessSubmissions']"}),
            'submission': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Submission']"}),
            'success': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'net.sipwcs': {
            'Meta': {'object_name': 'SipWCS'},
            'apterms': ('django.db.models.fields.TextField', [], {'default': "''"}),
            'aterms': ('django.db.models.fields.TextField', [], {'default': "''"}),
            'bpterms': ('django.db.models.fields.TextField', [], {'default': "''"}),
            'bterms': ('django.db.models.fields.TextField', [], {'default': "''"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'order': ('django.db.models.fields.PositiveSmallIntegerField', [], {'default': '2'}),
            'tan': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.TanWCS']", 'unique': 'True'})
        },
        'net.skylocation': {
            'Meta': {'object_name': 'SkyLocation'},
            'healpix': ('django.db.models.fields.BigIntegerField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'nside': ('django.db.models.fields.PositiveSmallIntegerField', [], {})
        },
        'net.skyobject': {
            'Meta': {'object_name': 'SkyObject'},
            'name': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'primary_key': 'True'})
        },
        'net.sourcelist': {
            'Meta': {'object_name': 'SourceList', '_ormbases': ['net.Image']},
            'image_ptr': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.Image']", 'unique': 'True', 'primary_key': 'True'}),
            'source_type': ('django.db.models.fields.CharField', [], {'max_length': '4'})
        },
        'net.submission': {
            'Meta': {'object_name': 'Submission'},
            'album': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Album']", 'null': 'True', 'blank': 'True'}),
            'center_dec': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'center_ra': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'comment_receiver': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.CommentReceiver']", 'unique': 'True'}),
            'deduplication_nonce': ('django.db.models.fields.IntegerField', [], {'null': 'True'}),
            'disk_file': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'submissions'", 'null': 'True', 'to': "orm['net.DiskFile']"}),
            'downsample_factor': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'error_message': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'license': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.License']"}),
            'original_filename': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'parity': ('django.db.models.fields.PositiveSmallIntegerField', [], {'default': '2'}),
            'positional_error': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'processing_finished': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'processing_retries': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'processing_started': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'publicly_visible': ('django.db.models.fields.CharField', [], {'default': "'y'", 'max_length': '1'}),
            'radius': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'scale_err': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'scale_est': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'scale_lower': ('django.db.models.fields.FloatField', [], {'default': '0.10000000000000001', 'null': 'True', 'blank': 'True'}),
            'scale_type': ('django.db.models.fields.CharField', [], {'default': "'ul'", 'max_length': '2'}),
            'scale_units': ('django.db.models.fields.CharField', [], {'default': "'degwidth'", 'max_length': '20'}),
            'scale_upper': ('django.db.models.fields.FloatField', [], {'default': '180', 'null': 'True', 'blank': 'True'}),
            'submitted_on': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'url': ('django.db.models.fields.URLField', [], {'max_length': '200', 'null': 'True', 'blank': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'submissions'", 'null': 'True', 'to': "orm['auth.User']"})
        },
        'net.tag': {
            'Meta': {'object_name': 'Tag'},
            'text': ('django.db.models.fields.CharField', [], {'max_length': '4096', 'primary_key': 'True'})
        },
        'net.taggeduserimage': {
            'Meta': {'object_name': 'TaggedUserImage'},
            'added_time': ('django.db.models.fields.DateTimeField', [], {'auto_now': 'True', 'blank': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'tag': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Tag']"}),
            'tagger': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['auth.User']", 'null': 'True'}),
            'user_image': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.UserImage']"})
        },
        'net.tanwcs': {
            'Meta': {'object_name': 'TanWCS'},
            'cd11': ('django.db.models.fields.FloatField', [], {}),
            'cd12': ('django.db.models.fields.FloatField', [], {}),
            'cd21': ('django.db.models.fields.FloatField', [], {}),
            'cd22': ('django.db.models.fields.FloatField', [], {}),
            'crpix1': ('django.db.models.fields.FloatField', [], {}),
            'crpix2': ('django.db.models.fields.FloatField', [], {}),
            'crval1': ('django.db.models.fields.FloatField', [], {}),
            'crval2': ('django.db.models.fields.FloatField', [], {}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'imageh': ('django.db.models.fields.FloatField', [], {}),
            'imagew': ('django.db.models.fields.FloatField', [], {})
        },
        'net.userimage': {
            'Meta': {'object_name': 'UserImage'},
            'comment_receiver': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.CommentReceiver']", 'unique': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '1024', 'blank': 'True'}),
            'flags': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'user_images'", 'symmetrical': 'False', 'through': "orm['net.FlaggedUserImage']", 'to': "orm['net.Flag']"}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'image': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Image']"}),
            'license': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.License']"}),
            'original_file_name': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'publicly_visible': ('django.db.models.fields.CharField', [], {'default': "'y'", 'max_length': '1'}),
            'sky_objects': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'user_images'", 'symmetrical': 'False', 'to': "orm['net.SkyObject']"}),
            'submission': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'user_images'", 'to': "orm['net.Submission']"}),
            'tags': ('django.db.models.fields.related.ManyToManyField', [], {'related_name': "'user_images'", 'symmetrical': 'False', 'through': "orm['net.TaggedUserImage']", 'to': "orm['net.Tag']"}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'user_images'", 'null': 'True', 'to': "orm['auth.User']"})
        },
        'net.userprofile': {
            'Meta': {'object_name': 'UserProfile'},
            'apikey': ('django.db.models.fields.CharField', [], {'max_length': '16'}),
            'default_license': ('django.db.models.fields.related.ForeignKey', [], {'default': '1', 'to': "orm['net.License']"}),
            'display_name': ('django.db.models.fields.CharField', [], {'max_length': '32'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'user': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'profile'", 'unique': 'True', 'to': "orm['auth.User']"})
        }
    }

    complete_apps = ['net']
