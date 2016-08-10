import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
import settings
from astrometry.net.models import *
from log import *



#for j in Job.objects.all():

for ui in UserImage.objects.all():
	jobs = ui.jobs.all()
	if len(jobs) <= 1:
		continue
	print('UserImage', ui.id, 'has jobs:', len(jobs))
	for j in jobs:
		print('  ', j.id, j.user_image.id)






#df = DiskFile.objects.get(file_hash='a03e51ca495bafe0a4db07b624200c0803d87d99')
def fix1():
	for df in DiskFile.objects.all().order_by('file_hash'):
		print(df)
		if os.path.exists(df.get_path()):
			#print 'exists'
			continue
	
		ocoll = df.collection
		for coll in ['cached', 'resized', 'uploaded', 'uploaded-gunzip', 'uploaded-untar']:
			df.collection = coll
			if os.path.exists(df.get_path()):
				print('--> found in', coll)
				df.save()
				break
	
