import time
import sys
import os.path
import threading

import astrometry.net.ice

## HACK -- the ICE auto-generated code doesn't seem to work well when it is
## embedded in another package.
sys.path.append(os.path.dirname(astrometry.net.ice.__file__))

from astrometry.net.portal.watcher_common import *

from astrometry.net.portal.watcher_script_ice import WatcherIce as watcherclass

from astrometry.net.portal.job import *
from astrometry.net.portal.queue import *
from astrometry.net.portal.log import log as logmsg

def runjob(qj):
	logmsg('Running job:', qj)
	w = watcherclass()
	if qj.job:
		w.handle_job(qj.job)
	else:
		w.run_sub(qj.sub)
	logmsg('QJob finished:', qj)
	logmsg('  job:', qj.job)
	logmsg('  sub:', qj.sub)
		
	requeue = False
	if qj.job:
		if qj.job.finished_without_error():
			qj.delete()
		else:
			requeue = True
	else:
		if qj.sub.finished_without_error() or qj.sub.alljobsadded:
			qj.delete()
		else:
			requeue = True
				
	# HACK - disable requeueing for now.
	# see also watcher_common.py:509
	if False and requeue:
		logmsg('Job failed: requeuing')
		qj.queuedtime = Job.timenow()
		qj.priority -= 1
		qj.ready = True
		qj.save()

		

def mainthread(nthreads):
	mythreads = []
	while True:
		#logmsg('Waiting for a thread to become available...')
		while True:
			for t in mythreads:
				#if not t.is_alive():
				if not t.isAlive():
					mythreads.remove(t)
					logmsg('Thread', t, 'finished.')
			if len(mythreads):
				logmsg('%i/%i threads running.' % (len(mythreads), nthreads))
			if len(mythreads) >= nthreads:
				time.sleep(10)
				continue
			break

		#logmsg('Getting next job...')
		nextjob = QueuedJob.next_job()
		logmsg('Next job:', nextjob);
		if nextjob is None:
			time.sleep(10)
			continue
		logmsg('Next job is', nextjob)

		t = threading.Thread(target=runjob, args=(nextjob,))
		mythreads.append(t)
		logmsg('Assigned job to thread', t)
		t.start()



if __name__ == '__main__':
	logmsg('Job queue:')
	for qj in QueuedJob.objects.all():
		logmsg('  ', qj)

	# DEBUG - enable one queued job.
	qjs = QueuedJob.objects.all()
	if len(qjs):
		logmsg('Enabling job', qjs[0])
		qjs[0].ready = True
		qjs[0].save()

	nthreads = 1

	mainthread(nthreads)
	
