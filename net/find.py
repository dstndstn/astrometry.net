import os
import sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)
import settings
from astrometry.net.models import *
from log import *


if __name__ == '__main__':
	import optparse
	parser = optparse.OptionParser('%(prog)')
	parser.add_option('-s', '--sub', type=int, dest='sub', help='Submission ID')
	parser.add_option('-j', '--job', type=int, dest='job', help='Job ID')
	parser.add_option('-u', '--userimage', type=int, dest='uimage', help='UserImage ID')
	parser.add_option('-r', '--rerun', dest='rerun', action='store_true',
					  help='Re-run this submission?')

	opt,args = parser.parse_args()
	if not (opt.sub or opt.job or opt.uimage):
		print 'Must specify one of --sub, --job, or --userimage'
		parser.print_help()
		sys.exit(-1)

	if opt.sub:
		sub = Submission.objects.all().get(id=opt.sub)
		print 'Submission', sub
		print 'Path', sub.disk_file.get_path()
		uis = sub.user_images.all()
		print 'UserImages:', len(uis)
		for ui in uis:
			print '	 ', ui
			print '	 with Jobs:', len(ui.jobs.all())
			for j in ui.jobs.all():
				print '	   ', j

		if opt.rerun:
			from process_submissions import try_dosub
			print 'Re-trying sub', sub.id
			try_dosub(sub, 1)

	if opt.job:
		job = Job.objects.all().get(id=opt.job)
		print 'Job', job
		print job.get_dir()
		ui = job.user_image
		print 'UserImage:', ui
		print 'User', ui.user
		im = ui.image
		print 'Image', im
		sub = ui.submission
		print 'Submission', sub
		print sub.disk_file.get_path()

	if opt.uimage:
		ui = UserImage.objects.all().get(id=opt.uimage)
		print 'UserImage', ui

		
	
