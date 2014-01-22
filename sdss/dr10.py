import os
import pyfits
from astrometry.util.fits import fits_table
import numpy as np

from common import *
from dr9 import *
from astrometry.util.yanny import *

class DR10(DR9):

	def __init__(self, **kwargs):
		'''
		Useful kwargs:
		
		basedir : (string) - local directory where data will be stored.
		'''
		DR9.__init__(self, **kwargs)
		self.dasurl = 'http://data.sdss3.org/sas/dr10/boss/'

	def getDRNumber(self):
		return 10
		
	def _get_runlist_filename(self):
		return self._get_data_file('runList-dr10.par')

