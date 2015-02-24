# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Album',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('publicly_visible', models.CharField(default=b'y', max_length=1, choices=[(b'y', b'yes'), (b'n', b'no')])),
                ('title', models.CharField(max_length=64)),
                ('description', models.CharField(max_length=1024, blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='CachedFile',
            fields=[
                ('key', models.CharField(max_length=64, unique=True, serialize=False, primary_key=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Calibration',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('ramin', models.FloatField()),
                ('ramax', models.FloatField()),
                ('decmin', models.FloatField()),
                ('decmax', models.FloatField()),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('z', models.FloatField()),
                ('r', models.FloatField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Comment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('text', models.CharField(max_length=1024)),
                ('author', models.ForeignKey(related_name='comments_left', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='CommentReceiver',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('owner', models.ForeignKey(to=settings.AUTH_USER_MODEL, null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='DiskFile',
            fields=[
                ('collection', models.CharField(default=b'misc', max_length=40)),
                ('file_hash', models.CharField(max_length=40, unique=True, serialize=False, primary_key=True)),
                ('size', models.PositiveIntegerField()),
                ('file_type', models.CharField(max_length=256, null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='EnhancedImage',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('nside', models.IntegerField()),
                ('healpix', models.IntegerField()),
                ('maxweight', models.FloatField(default=0.0)),
                ('cals', models.ManyToManyField(related_name='enhanced_images', db_table=b'enhancedimage_calibration', to='net.Calibration')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='EnhanceVersion',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=64)),
                ('topscale', models.FloatField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Flag',
            fields=[
                ('name', models.CharField(max_length=56, serialize=False, primary_key=True)),
                ('explanation', models.CharField(max_length=2048, blank=True)),
            ],
            options={
                'ordering': ['name'],
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='FlaggedUserImage',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('flagged_time', models.DateTimeField(auto_now=True)),
                ('flag', models.ForeignKey(to='net.Flag')),
                ('user', models.ForeignKey(to=settings.AUTH_USER_MODEL)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('width', models.PositiveIntegerField(null=True)),
                ('height', models.PositiveIntegerField(null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Job',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('status', models.CharField(max_length=1, choices=[(b'S', b'Success'), (b'F', b'Failure')])),
                ('error_message', models.CharField(max_length=256)),
                ('queued_time', models.DateTimeField(null=True)),
                ('start_time', models.DateTimeField(null=True)),
                ('end_time', models.DateTimeField(null=True)),
                ('calibration', models.OneToOneField(related_name='job', null=True, to='net.Calibration')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='License',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('allow_commercial_use', models.CharField(default=b'd', max_length=1, choices=[(b'y', b'yes'), (b'n', b'no'), (b'd', b'use default')])),
                ('allow_modifications', models.CharField(default=b'd', max_length=2, choices=[(b'y', b'yes'), (b'sa', b'yes, but share alike'), (b'n', b'no'), (b'd', b'use default')])),
                ('license_name', models.CharField(max_length=1024)),
                ('license_uri', models.CharField(max_length=1024)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='ProcessSubmissions',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('pid', models.IntegerField()),
                ('watchdog', models.DateTimeField(null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='QueuedJob',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('finished', models.BooleanField(default=False)),
                ('success', models.BooleanField(default=False)),
                ('job', models.ForeignKey(to='net.Job')),
                ('procsub', models.ForeignKey(related_name='jobs', to='net.ProcessSubmissions')),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='QueuedSubmission',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('finished', models.BooleanField(default=False)),
                ('success', models.BooleanField(default=False)),
                ('procsub', models.ForeignKey(related_name='subs', to='net.ProcessSubmissions')),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='SipWCS',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('order', models.PositiveSmallIntegerField(default=2)),
                ('aterms', models.TextField(default=b'')),
                ('bterms', models.TextField(default=b'')),
                ('apterms', models.TextField(default=b'')),
                ('bpterms', models.TextField(default=b'')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='SkyLocation',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('nside', models.PositiveSmallIntegerField()),
                ('healpix', models.BigIntegerField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='SkyObject',
            fields=[
                ('name', models.CharField(max_length=1024, serialize=False, primary_key=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='SourceList',
            fields=[
                ('image_ptr', models.OneToOneField(parent_link=True, auto_created=True, primary_key=True, serialize=False, to='net.Image')),
                ('source_type', models.CharField(max_length=4, choices=[(b'fits', b'FITS binary table'), (b'text', b'Text list')])),
            ],
            options={
            },
            bases=('net.image',),
        ),
        migrations.CreateModel(
            name='Submission',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('publicly_visible', models.CharField(default=b'y', max_length=1, choices=[(b'y', b'yes'), (b'n', b'no')])),
                ('url', models.URLField(null=True, blank=True)),
                ('parity', models.PositiveSmallIntegerField(default=2, choices=[(2, b'try both simultaneously'), (0, b'positive'), (1, b'negative')])),
                ('scale_units', models.CharField(default=b'degwidth', max_length=20, choices=[(b'arcsecperpix', b'arcseconds per pixel'), (b'arcminwidth', b'width of the field (in arcminutes)'), (b'degwidth', b'width of the field (in degrees)'), (b'focalmm', b'focal length of the lens (for 35mm film equivalent sensor)')])),
                ('scale_type', models.CharField(default=b'ul', max_length=2, choices=[(b'ul', b'bounds'), (b'ev', b'estimate +/- error')])),
                ('scale_lower', models.FloatField(default=0.1, null=True, blank=True)),
                ('scale_upper', models.FloatField(default=180, null=True, blank=True)),
                ('scale_est', models.FloatField(null=True, blank=True)),
                ('scale_err', models.FloatField(null=True, blank=True)),
                ('positional_error', models.FloatField(null=True, blank=True)),
                ('center_ra', models.FloatField(null=True, blank=True)),
                ('center_dec', models.FloatField(null=True, blank=True)),
                ('radius', models.FloatField(null=True, blank=True)),
                ('tweak_order', models.IntegerField(default=2, null=True, blank=True)),
                ('downsample_factor', models.PositiveIntegerField(default=2, null=True, blank=True)),
                ('use_sextractor', models.BooleanField(default=False)),
                ('crpix_center', models.BooleanField(default=False)),
                ('invert', models.BooleanField(default=False)),
                ('image_width', models.IntegerField(default=0, null=True, blank=True)),
                ('image_height', models.IntegerField(default=0, null=True, blank=True)),
                ('via_api', models.BooleanField(default=False)),
                ('original_filename', models.CharField(max_length=256)),
                ('submitted_on', models.DateTimeField(auto_now_add=True)),
                ('processing_started', models.DateTimeField(null=True)),
                ('processing_finished', models.DateTimeField(null=True)),
                ('processing_retries', models.PositiveIntegerField(default=0)),
                ('error_message', models.CharField(max_length=2048, null=True)),
                ('album', models.ForeignKey(blank=True, to='net.Album', null=True)),
                ('comment_receiver', models.OneToOneField(to='net.CommentReceiver')),
                ('disk_file', models.ForeignKey(related_name='submissions', to='net.DiskFile', null=True)),
                ('license', models.ForeignKey(to='net.License')),
                ('user', models.ForeignKey(related_name='submissions', to=settings.AUTH_USER_MODEL, null=True)),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('text', models.CharField(max_length=4096, serialize=False, primary_key=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='TaggedUserImage',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('added_time', models.DateTimeField(auto_now=True)),
                ('tag', models.ForeignKey(to='net.Tag')),
                ('tagger', models.ForeignKey(to=settings.AUTH_USER_MODEL, null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='TanWCS',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('crval1', models.FloatField()),
                ('crval2', models.FloatField()),
                ('crpix1', models.FloatField()),
                ('crpix2', models.FloatField()),
                ('cd11', models.FloatField()),
                ('cd12', models.FloatField()),
                ('cd21', models.FloatField()),
                ('cd22', models.FloatField()),
                ('imagew', models.FloatField()),
                ('imageh', models.FloatField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='UserImage',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('publicly_visible', models.CharField(default=b'y', max_length=1, choices=[(b'y', b'yes'), (b'n', b'no')])),
                ('description', models.CharField(max_length=1024, blank=True)),
                ('original_file_name', models.CharField(max_length=256)),
                ('comment_receiver', models.OneToOneField(to='net.CommentReceiver')),
                ('flags', models.ManyToManyField(related_name='user_images', through='net.FlaggedUserImage', to='net.Flag')),
                ('image', models.ForeignKey(to='net.Image')),
                ('license', models.ForeignKey(to='net.License')),
                ('sky_objects', models.ManyToManyField(related_name='user_images', to='net.SkyObject')),
                ('submission', models.ForeignKey(related_name='user_images', to='net.Submission')),
                ('tags', models.ManyToManyField(related_name='user_images', through='net.TaggedUserImage', to='net.Tag')),
                ('user', models.ForeignKey(related_name='user_images', to=settings.AUTH_USER_MODEL, null=True)),
            ],
            options={
                'abstract': False,
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('display_name', models.CharField(max_length=32)),
                ('apikey', models.CharField(max_length=16)),
                ('default_license', models.ForeignKey(default=1, to='net.License')),
                ('user', models.ForeignKey(related_name='profile', editable=False, to=settings.AUTH_USER_MODEL, unique=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='taggeduserimage',
            name='user_image',
            field=models.ForeignKey(to='net.UserImage'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='sipwcs',
            name='tan',
            field=models.OneToOneField(to='net.TanWCS'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='queuedsubmission',
            name='submission',
            field=models.ForeignKey(to='net.Submission'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='job',
            name='user_image',
            field=models.ForeignKey(related_name='jobs', to='net.UserImage'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='image',
            name='disk_file',
            field=models.ForeignKey(to='net.DiskFile'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='image',
            name='display_image',
            field=models.ForeignKey(related_name='image_display_set', to='net.Image', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='image',
            name='thumbnail',
            field=models.ForeignKey(related_name='image_thumbnail_set', to='net.Image', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='flaggeduserimage',
            name='user_image',
            field=models.ForeignKey(to='net.UserImage'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='enhancedimage',
            name='version',
            field=models.ForeignKey(to='net.EnhanceVersion'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='enhancedimage',
            name='wcs',
            field=models.ForeignKey(default=None, to='net.TanWCS', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='comment',
            name='recipient',
            field=models.ForeignKey(related_name='comments', to='net.CommentReceiver'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='calibration',
            name='raw_tan',
            field=models.ForeignKey(related_name='calibrations_raw', to='net.TanWCS', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='calibration',
            name='sip',
            field=models.ForeignKey(to='net.SipWCS', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='calibration',
            name='sky_location',
            field=models.ForeignKey(related_name='calibrations', to='net.SkyLocation', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='calibration',
            name='tweaked_tan',
            field=models.ForeignKey(related_name='calibrations_tweaked', to='net.TanWCS', null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='cachedfile',
            name='disk_file',
            field=models.ForeignKey(to='net.DiskFile'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='album',
            name='comment_receiver',
            field=models.OneToOneField(to='net.CommentReceiver'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='album',
            name='tags',
            field=models.ManyToManyField(related_name='albums', to='net.Tag'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='album',
            name='user',
            field=models.ForeignKey(related_name='albums', to=settings.AUTH_USER_MODEL, null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='album',
            name='user_images',
            field=models.ManyToManyField(related_name='albums', to='net.UserImage'),
            preserve_default=True,
        ),
    ]
