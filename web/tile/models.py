from django.db import models

from an.util import orderby
import logging

class Image(models.Model):
	objects = orderby.OrderByManager()
	
	# The original image filename.
	origfilename = models.CharField(max_length=1024)
	# The original image file format.
	origformat = models.CharField(max_length=30)
	# The JPEG base filename.
	filename = models.CharField(max_length=1024)
	ramin =	 models.FloatField()
	ramax =	 models.FloatField()
	decmin = models.FloatField()
	decmax = models.FloatField()
	imagew = models.IntegerField()
	imageh = models.IntegerField()

	def __str__(self):
		return self.origfilename

