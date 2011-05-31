from math import floor, ceil

from django.db import models
from django.db.models import Q

from astrometry.net1.util import orderby
from astrometry.net1.portal.log import log as logmsg
from astrometry.net1.portal.job import Job

class MapImage(models.Model):
	objects = orderby.OrderByManager()

	# The original image filename.
	#origfilename = models.CharField(max_length=1024)
	# The original image file format.
	#origformat = models.CharField(max_length=30)
	# The JPEG base filename.
	#filename = models.CharField(max_length=1024)

	job = models.ForeignKey(Job, related_name='render', unique=True)

	# have we already tried to get the EXIF date?
	checked_exif_date = models.BooleanField()
	# exif tag "EXIF DateTimeOriginal"
	exif_orig_date = models.DateTimeField(null=True)
	# exif tag "Image DateTime"
	exif_date = models.DateTimeField(null=True)

	# cached from job.calibration.*
	ramin =	 models.FloatField()
	ramax =	 models.FloatField()
	decmin = models.FloatField()
	decmax = models.FloatField()

	def __str__(self):
		return self.get_orig_filename()

	def get_date(self):
		return (self.exif_orig_date or self.exif_date) or None

	def check_exif_date(self):
		from astrometry.util import EXIF
		from datetime import datetime

		if self.checked_exif_date:
			return
		if not self.job.diskfile:
			raise 'No diskfile?'
		format = '%Y:%m:%d %H:%M:%S'

		p = self.job.diskfile.get_path()
		tags = EXIF.process_file(open(p))

		t = tags.get('EXIF DateTimeOriginal')
		if t:
			#logmsg('File', p, 'orig time:', t)
			try:
				thetime = datetime.strptime(str(t), format)
				if thetime:
					self.exif_orig_date = thetime
			except Exception,ex:
				pass
		t = tags.get('Image DateTime')
		if t:
			#logmsg('File', p, 'time:', t)
			try:
				thetime = datetime.strptime(str(t), format)
				if thetime:
					self.exif_date = thetime
			except Exception,ex:
				pass

		self.checked_exif_date = True
		self.save()

	def get_orig_filename(self):
		return self.job.get_orig_file()

	@staticmethod
	def have_dates(queryset):
		Qorig = Q(exif_orig_date__isnull=False)
		Qdate = Q(exif_date__isnull=False)
		return queryset.filter(Qorig | Qdate)

class MapImageSet(models.Model):
	images = models.ManyToManyField(MapImage, related_name='imagesets')

	datelo = models.DateTimeField(null=True)
	datehi = models.DateTimeField(null=True)

	def cache_dates(self):
	   	tocache = self.images.filter(checked_exif_date=False)
		for img in tocache:
			img.check_exif_date()

	def get_date_range(self):
		if self.datelo and self.datehi:
			return (self.datelo, self.datehi)
		self.cache_dates()

		# logmsg("%i dates" % len(dates))
		# logmsg("earliest:", min(dates))
		# logmsg("latest:", max(dates))

		# percentile clipping
		pct = 5.
		havedates = MapImage.have_dates(self.images)
		n = havedates.count()
		logmsg('MapImageSet', self.id, 'has %i images with dates' % n)
		ilo = int(ceil(n * pct / 100.))
		ihi = int(floor(n * (1. - pct / 100.)))

		# FIXME - could perhaps do this in SQL...
		dates = [image.get_date() for image in havedates]
		dates.sort()
		self.datelo = dates[ilo]
		self.datehi = dates[ihi]
		self.save()
		return (self.datelo, self.datehi)

