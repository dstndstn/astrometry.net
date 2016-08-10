from __future__ import print_function
import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
import settings
from astrometry.net.models import *
from log import *

if __name__ == '__main__':
	import optparse
	parser = optparse.OptionParser('%(prog) <file>')
	opt,args = parser.parse_args()
	if len(args) != 1:
		parser.print_usage()
		sys.exit(-1)

	fn = args[0]

	df = DiskFile.from_file(fn)
	df.set_size_and_file_type()
	df.save()
	print('Made DiskFile', df)

	# EVIL
	#Submission.objects.all().delete()
	
	sub = Submission(disk_file=df, scale_type='ul', scale_units='degwidth')
	sub.original_filename = fn
	sub.save()
	print('Made Submission', sub)
