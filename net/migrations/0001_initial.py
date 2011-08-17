# encoding: utf-8
import datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models

class Migration(SchemaMigration):

    def forwards(self, orm):
        
        # Adding model 'TanWCS'
        db.create_table('net_tanwcs', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('crval1', self.gf('django.db.models.fields.FloatField')()),
            ('crval2', self.gf('django.db.models.fields.FloatField')()),
            ('crpix1', self.gf('django.db.models.fields.FloatField')()),
            ('crpix2', self.gf('django.db.models.fields.FloatField')()),
            ('cd11', self.gf('django.db.models.fields.FloatField')()),
            ('cd12', self.gf('django.db.models.fields.FloatField')()),
            ('cd21', self.gf('django.db.models.fields.FloatField')()),
            ('cd22', self.gf('django.db.models.fields.FloatField')()),
            ('imagew', self.gf('django.db.models.fields.FloatField')()),
            ('imageh', self.gf('django.db.models.fields.FloatField')()),
        ))
        db.send_create_signal('net', ['TanWCS'])

        # Adding model 'SipWCS'
        db.create_table('net_sipwcs', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('tan', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.TanWCS'], unique=True)),
            ('order', self.gf('django.db.models.fields.PositiveSmallIntegerField')(default=2)),
            ('aterms', self.gf('django.db.models.fields.TextField')(default='')),
            ('bterms', self.gf('django.db.models.fields.TextField')(default='')),
            ('apterms', self.gf('django.db.models.fields.TextField')(default='')),
            ('bpterms', self.gf('django.db.models.fields.TextField')(default='')),
        ))
        db.send_create_signal('net', ['SipWCS'])

        # Adding model 'License'
        db.create_table('net_license', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('allow_commercial_use', self.gf('django.db.models.fields.CharField')(default='d', max_length=1)),
            ('allow_modifications', self.gf('django.db.models.fields.CharField')(default='d', max_length=2)),
            ('license_name', self.gf('django.db.models.fields.CharField')(max_length=1024)),
            ('license_uri', self.gf('django.db.models.fields.CharField')(max_length=1024)),
        ))
        db.send_create_signal('net', ['License'])

        # Adding model 'CommentReceiver'
        db.create_table('net_commentreceiver', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('owner', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['auth.User'], null=True)),
        ))
        db.send_create_signal('net', ['CommentReceiver'])

        # Adding model 'ProcessSubmissions'
        db.create_table('net_processsubmissions', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('pid', self.gf('django.db.models.fields.IntegerField')()),
            ('watchdog', self.gf('django.db.models.fields.DateTimeField')(null=True)),
        ))
        db.send_create_signal('net', ['ProcessSubmissions'])

        # Adding model 'QueuedSubmission'
        db.create_table('net_queuedsubmission', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('finished', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('success', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('procsub', self.gf('django.db.models.fields.related.ForeignKey')(related_name='subs', to=orm['net.ProcessSubmissions'])),
            ('submission', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.Submission'])),
        ))
        db.send_create_signal('net', ['QueuedSubmission'])

        # Adding model 'QueuedJob'
        db.create_table('net_queuedjob', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('finished', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('success', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('procsub', self.gf('django.db.models.fields.related.ForeignKey')(related_name='jobs', to=orm['net.ProcessSubmissions'])),
            ('job', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.Job'])),
        ))
        db.send_create_signal('net', ['QueuedJob'])

        # Adding model 'DiskFile'
        db.create_table('net_diskfile', (
            ('file_hash', self.gf('django.db.models.fields.CharField')(unique=True, max_length=40, primary_key=True)),
            ('size', self.gf('django.db.models.fields.PositiveIntegerField')()),
            ('file_type', self.gf('django.db.models.fields.CharField')(max_length=256, null=True)),
        ))
        db.send_create_signal('net', ['DiskFile'])

        # Adding model 'CachedFile'
        db.create_table('net_cachedfile', (
            ('disk_file', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.DiskFile'])),
            ('key', self.gf('django.db.models.fields.CharField')(unique=True, max_length=64, primary_key=True)),
        ))
        db.send_create_signal('net', ['CachedFile'])

        # Adding model 'Image'
        db.create_table('net_image', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('disk_file', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.DiskFile'])),
            ('width', self.gf('django.db.models.fields.PositiveIntegerField')(null=True)),
            ('height', self.gf('django.db.models.fields.PositiveIntegerField')(null=True)),
            ('thumbnail', self.gf('django.db.models.fields.related.ForeignKey')(related_name='image_thumbnail_set', null=True, to=orm['net.Image'])),
            ('display_image', self.gf('django.db.models.fields.related.ForeignKey')(related_name='image_display_set', null=True, to=orm['net.Image'])),
        ))
        db.send_create_signal('net', ['Image'])

        # Adding model 'SourceList'
        db.create_table('net_sourcelist', (
            ('image_ptr', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.Image'], unique=True, primary_key=True)),
            ('source_type', self.gf('django.db.models.fields.CharField')(max_length=4)),
        ))
        db.send_create_signal('net', ['SourceList'])

        # Adding model 'Tag'
        db.create_table('net_tag', (
            ('text', self.gf('django.db.models.fields.CharField')(max_length=4096, primary_key=True)),
        ))
        db.send_create_signal('net', ['Tag'])

        # Adding model 'Calibration'
        db.create_table('net_calibration', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('raw_tan', self.gf('django.db.models.fields.related.ForeignKey')(related_name='calibrations_raw', null=True, to=orm['net.TanWCS'])),
            ('tweaked_tan', self.gf('django.db.models.fields.related.ForeignKey')(related_name='calibrations_tweaked', null=True, to=orm['net.TanWCS'])),
            ('sip', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.SipWCS'], null=True)),
            ('ramin', self.gf('django.db.models.fields.FloatField')()),
            ('ramax', self.gf('django.db.models.fields.FloatField')()),
            ('decmin', self.gf('django.db.models.fields.FloatField')()),
            ('decmax', self.gf('django.db.models.fields.FloatField')()),
            ('sky_location', self.gf('django.db.models.fields.related.ForeignKey')(related_name='calibrations', null=True, to=orm['net.SkyLocation'])),
        ))
        db.send_create_signal('net', ['Calibration'])

        # Adding model 'Job'
        db.create_table('net_job', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('calibration', self.gf('django.db.models.fields.related.ForeignKey')(related_name='jobs', null=True, to=orm['net.Calibration'])),
            ('status', self.gf('django.db.models.fields.CharField')(max_length=1)),
            ('error_message', self.gf('django.db.models.fields.CharField')(max_length=256)),
            ('user_image', self.gf('django.db.models.fields.related.ForeignKey')(related_name='jobs', to=orm['net.UserImage'])),
            ('queued_time', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            ('start_time', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            ('end_time', self.gf('django.db.models.fields.DateTimeField')(null=True)),
        ))
        db.send_create_signal('net', ['Job'])

        # Adding model 'SkyLocation'
        db.create_table('net_skylocation', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('nside', self.gf('django.db.models.fields.PositiveSmallIntegerField')()),
            ('healpix', self.gf('django.db.models.fields.BigIntegerField')()),
        ))
        db.send_create_signal('net', ['SkyLocation'])

        # Adding model 'TaggedUserImage'
        db.create_table('net_taggeduserimage', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('user_image', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.UserImage'])),
            ('tag', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.Tag'])),
            ('tagger', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['auth.User'], null=True)),
            ('added_time', self.gf('django.db.models.fields.DateTimeField')(auto_now=True, blank=True)),
        ))
        db.send_create_signal('net', ['TaggedUserImage'])

        # Adding model 'UserImage'
        db.create_table('net_userimage', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('publicly_visible', self.gf('django.db.models.fields.CharField')(default='y', max_length=1)),
            ('image', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.Image'])),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(related_name='user_images', null=True, to=orm['auth.User'])),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('original_file_name', self.gf('django.db.models.fields.CharField')(max_length=256)),
            ('submission', self.gf('django.db.models.fields.related.ForeignKey')(related_name='user_images', to=orm['net.Submission'])),
            ('license', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.License'], unique=True)),
            ('comment_receiver', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.CommentReceiver'], unique=True)),
        ))
        db.send_create_signal('net', ['UserImage'])

        # Adding model 'Submission'
        db.create_table('net_submission', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('publicly_visible', self.gf('django.db.models.fields.CharField')(default='y', max_length=1)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(related_name='submissions', null=True, to=orm['auth.User'])),
            ('disk_file', self.gf('django.db.models.fields.related.ForeignKey')(related_name='submissions', null=True, to=orm['net.DiskFile'])),
            ('url', self.gf('django.db.models.fields.URLField')(max_length=200, null=True, blank=True)),
            ('parity', self.gf('django.db.models.fields.PositiveSmallIntegerField')(default=2)),
            ('scale_units', self.gf('django.db.models.fields.CharField')(default='degwidth', max_length=20)),
            ('scale_type', self.gf('django.db.models.fields.CharField')(default='ul', max_length=2)),
            ('scale_lower', self.gf('django.db.models.fields.FloatField')(default=0.10000000000000001, null=True, blank=True)),
            ('scale_upper', self.gf('django.db.models.fields.FloatField')(default=180, null=True, blank=True)),
            ('scale_est', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('scale_err', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('positional_error', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('center_ra', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('center_dec', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('radius', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('downsample_factor', self.gf('django.db.models.fields.PositiveIntegerField')(null=True, blank=True)),
            ('source_type', self.gf('django.db.models.fields.CharField')(default='image', max_length=5)),
            ('original_filename', self.gf('django.db.models.fields.CharField')(max_length=256)),
            ('album', self.gf('django.db.models.fields.related.ForeignKey')(to=orm['net.Album'], null=True, blank=True)),
            ('submitted_on', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('processing_started', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            ('processing_finished', self.gf('django.db.models.fields.DateTimeField')(null=True)),
            ('error_message', self.gf('django.db.models.fields.CharField')(max_length=256, null=True)),
            ('license', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.License'], unique=True)),
            ('comment_receiver', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.CommentReceiver'], unique=True)),
        ))
        db.send_create_signal('net', ['Submission'])

        # Adding model 'Album'
        db.create_table('net_album', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('publicly_visible', self.gf('django.db.models.fields.CharField')(default='y', max_length=1)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(related_name='albums', null=True, to=orm['auth.User'])),
            ('title', self.gf('django.db.models.fields.CharField')(max_length=64)),
            ('description', self.gf('django.db.models.fields.CharField')(max_length=1024, blank=True)),
            ('created_at', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('comment_receiver', self.gf('django.db.models.fields.related.OneToOneField')(to=orm['net.CommentReceiver'], unique=True)),
        ))
        db.send_create_signal('net', ['Album'])

        # Adding M2M table for field user_images on 'Album'
        db.create_table('net_album_user_images', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('album', models.ForeignKey(orm['net.album'], null=False)),
            ('userimage', models.ForeignKey(orm['net.userimage'], null=False))
        ))
        db.create_unique('net_album_user_images', ['album_id', 'userimage_id'])

        # Adding M2M table for field tags on 'Album'
        db.create_table('net_album_tags', (
            ('id', models.AutoField(verbose_name='ID', primary_key=True, auto_created=True)),
            ('album', models.ForeignKey(orm['net.album'], null=False)),
            ('tag', models.ForeignKey(orm['net.tag'], null=False))
        ))
        db.create_unique('net_album_tags', ['album_id', 'tag_id'])

        # Adding model 'Comment'
        db.create_table('net_comment', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('created_at', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('recipient', self.gf('django.db.models.fields.related.ForeignKey')(related_name='comments', to=orm['net.CommentReceiver'])),
            ('author', self.gf('django.db.models.fields.related.ForeignKey')(related_name='comments_left', to=orm['auth.User'])),
            ('text', self.gf('django.db.models.fields.CharField')(max_length=1024)),
        ))
        db.send_create_signal('net', ['Comment'])

        # Adding model 'UserProfile'
        db.create_table('net_userprofile', (
            ('id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('display_name', self.gf('django.db.models.fields.CharField')(max_length=32)),
            ('user', self.gf('django.db.models.fields.related.ForeignKey')(related_name='profile', unique=True, to=orm['auth.User'])),
            ('apikey', self.gf('django.db.models.fields.CharField')(max_length=16)),
            ('default_license', self.gf('django.db.models.fields.related.ForeignKey')(default=1, to=orm['net.License'])),
        ))
        db.send_create_signal('net', ['UserProfile'])


    def backwards(self, orm):
        
        # Deleting model 'TanWCS'
        db.delete_table('net_tanwcs')

        # Deleting model 'SipWCS'
        db.delete_table('net_sipwcs')

        # Deleting model 'License'
        db.delete_table('net_license')

        # Deleting model 'CommentReceiver'
        db.delete_table('net_commentreceiver')

        # Deleting model 'ProcessSubmissions'
        db.delete_table('net_processsubmissions')

        # Deleting model 'QueuedSubmission'
        db.delete_table('net_queuedsubmission')

        # Deleting model 'QueuedJob'
        db.delete_table('net_queuedjob')

        # Deleting model 'DiskFile'
        db.delete_table('net_diskfile')

        # Deleting model 'CachedFile'
        db.delete_table('net_cachedfile')

        # Deleting model 'Image'
        db.delete_table('net_image')

        # Deleting model 'SourceList'
        db.delete_table('net_sourcelist')

        # Deleting model 'Tag'
        db.delete_table('net_tag')

        # Deleting model 'Calibration'
        db.delete_table('net_calibration')

        # Deleting model 'Job'
        db.delete_table('net_job')

        # Deleting model 'SkyLocation'
        db.delete_table('net_skylocation')

        # Deleting model 'TaggedUserImage'
        db.delete_table('net_taggeduserimage')

        # Deleting model 'UserImage'
        db.delete_table('net_userimage')

        # Deleting model 'Submission'
        db.delete_table('net_submission')

        # Deleting model 'Album'
        db.delete_table('net_album')

        # Removing M2M table for field user_images on 'Album'
        db.delete_table('net_album_user_images')

        # Removing M2M table for field tags on 'Album'
        db.delete_table('net_album_tags')

        # Deleting model 'Comment'
        db.delete_table('net_comment')

        # Deleting model 'UserProfile'
        db.delete_table('net_userprofile')


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
            'ramax': ('django.db.models.fields.FloatField', [], {}),
            'ramin': ('django.db.models.fields.FloatField', [], {}),
            'raw_tan': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'calibrations_raw'", 'null': 'True', 'to': "orm['net.TanWCS']"}),
            'sip': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.SipWCS']", 'null': 'True'}),
            'sky_location': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'calibrations'", 'null': 'True', 'to': "orm['net.SkyLocation']"}),
            'tweaked_tan': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'calibrations_tweaked'", 'null': 'True', 'to': "orm['net.TanWCS']"})
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
            'calibration': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'jobs'", 'null': 'True', 'to': "orm['net.Calibration']"}),
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
            'disk_file': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'submissions'", 'null': 'True', 'to': "orm['net.DiskFile']"}),
            'downsample_factor': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'error_message': ('django.db.models.fields.CharField', [], {'max_length': '256', 'null': 'True'}),
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'license': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.License']", 'unique': 'True'}),
            'original_filename': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'parity': ('django.db.models.fields.PositiveSmallIntegerField', [], {'default': '2'}),
            'positional_error': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'processing_finished': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'processing_started': ('django.db.models.fields.DateTimeField', [], {'null': 'True'}),
            'publicly_visible': ('django.db.models.fields.CharField', [], {'default': "'y'", 'max_length': '1'}),
            'radius': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'scale_err': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'scale_est': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'scale_lower': ('django.db.models.fields.FloatField', [], {'default': '0.10000000000000001', 'null': 'True', 'blank': 'True'}),
            'scale_type': ('django.db.models.fields.CharField', [], {'default': "'ul'", 'max_length': '2'}),
            'scale_units': ('django.db.models.fields.CharField', [], {'default': "'degwidth'", 'max_length': '20'}),
            'scale_upper': ('django.db.models.fields.FloatField', [], {'default': '180', 'null': 'True', 'blank': 'True'}),
            'source_type': ('django.db.models.fields.CharField', [], {'default': "'image'", 'max_length': '5'}),
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
            'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'image': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['net.Image']"}),
            'license': ('django.db.models.fields.related.OneToOneField', [], {'to': "orm['net.License']", 'unique': 'True'}),
            'original_file_name': ('django.db.models.fields.CharField', [], {'max_length': '256'}),
            'publicly_visible': ('django.db.models.fields.CharField', [], {'default': "'y'", 'max_length': '1'}),
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
