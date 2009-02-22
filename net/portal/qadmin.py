#! /usr/bin/env python

import astrometry.net.django_commandline

import sys

from astrometry.net.portal.queue import *
from astrometry.net.portal.job import *

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print 'Usage: qadmin.py <action> [args]'
		print '	 <action> can be:'
		print '		"clear" - remove everything from the queue.'
		print '		"show"	- show the queue.'
		print '		"ready" - make all jobs ready.'
		print '			[n] - make "n" jobs ready'
		print '		"requeue <job or sub id>" - reset all to "Queued" status'
		sys.exit(-1)

	action = sys.argv[1]
	if action == 'show':
		qjs = QueuedJob.objects.all()
		print '%i queued jobs:' % qjs.count()
		for qj in qjs:
			print '	 ', qj

	elif action == 'ready':
		n = -1
		if len(sys.argv) >= 3:
			n = int(sys.argv[2])
		qjs = QueuedJob.objects.all().filter(ready=False)
		if n > 0:
			qjs = qjs[:n]
		for qj in qjs:
			qj.ready = True
			if qj.job:
				qj.job.set_status('Queued')
			if qj.sub:
				qj.sub.set_status('Queued')
			qj.save()
			print 'Made ready:', qj

	elif action == 'clear':
		print 'Deleting all queued jobs...'
		QueuedJob.objects.all().delete()

	elif action == 'requeue':
		if len(sys.argv) < 3:
			print 'Need job or submission id.'
			sys.exit(-1)
		jid = sys.argv[2]
		subs = Submission.objects.filter(subid=jid)
		if subs.count():
			jobs = subs[0].jobs.all()
		else:
			jobs = Job.objects.filter(jobid=jid)
		jobs = list(jobs)
		print '%i jobs.' % len(jobs)
		for j in jobs:
			print '	 ', j.jobid
			j.set_status('Queued')
			j.save()
			QueuedJob.submit_job(j)
		
