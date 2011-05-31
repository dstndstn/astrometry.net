import random

from django.db import models
from django.db import transaction
from django.contrib.auth.models import User, AnonymousUser

#from astrometry.net1.portal.job import Job,Submission
from astrometry.net1.portal.job import *
from astrometry.net1.portal.log import log as logmsg

class QueuedJob(models.Model):
	priority_high = 1500
	priority_normal = 1000
	priority_low = 500

	job = models.ForeignKey('Job', related_name='queued', null=True)
	sub = models.ForeignKey('Submission', related_name='queued', null=True)
	# derived from job.user or sub.user for query efficiency.
	user = models.ForeignKey(User, related_name='user', null=True)

	queuedtime = models.DateTimeField()
	priority = models.PositiveIntegerField()
	ready = models.BooleanField()

	def __str__(self):
		elems = []
		if self.job:
			elems.append('job=' + self.job.jobid)
		if self.sub:
			elems.append('sub=' + self.sub.subid)
		if self.ready:
			elems.append('ready')
		else:
			elems.append('not ready')
		return '<QJ: ' + ', '.join(elems) + '>'

	# how many times have we previously attempted to run this job?
	def count_failures(self):
		return self.runs.count()

	def has_failed_before(self):
		return self.count_failures() > 0

	#@transaction.commit_on_success
	@staticmethod
	def submit_job(job, priority=priority_normal):
		from astrometry.net1.portal.job import Job,Submission
		qj = QueuedJob(job=job,
					   sub=None,
					   user=job.get_user(),
					   queuedtime=Job.timenow(),
					   priority=priority,
					   ready=True)
		qj.save()


	@staticmethod
	#@transaction.commit_on_success
	def submit_submission(sub, priority=priority_normal):
		from astrometry.net1.portal.job import Job,Submission
		qj = QueuedJob(sub=sub,
					   job=None,
					   user=sub.user,
					   queuedtime=Job.timenow(),
					   priority=priority,
					   ready=True)
		qj.save()

	@staticmethod
	def submit_job_or_submission(js, priority=priority_normal):
		from astrometry.net1.portal.job import Job,Submission
		if isinstance(js, Job):
			QueuedJob.submit_job(js, priority)
		elif isinstance(js, Submission):
			QueuedJob.submit_submission(js, priority)
		else:
			logmsg('Neither job not submission in submit_job_or_submission')
			raise TypeError('neither job nor submission:', js)

	#@transaction.commit_on_success
	@staticmethod
	def next_job():
		q = QueuedJob.objects.filter(ready=True).values_list('priority', flat=True).order_by('-priority')
		if q.count() == 0:
			return None
		toppriority = q[0]
		logmsg('Top priority is', toppriority)
		#toppriority = q.priority
		#users = QueuedJob.objects.filter(ready=True, priority=toppriority).values_list('job_user', 'sub_user')
		userids = QueuedJob.objects.filter(ready=True, priority=toppriority).values_list('user', flat=True).distinct()
		logmsg('User ids of users with queued jobs:', userids)
		# debug
		if True:
			users = [uid and User.objects.get(id=uid) or AnonymousUser() for uid in userids]
			logmsg('Users with queued jobs:', users)
		### NOTE, random.randint and numpy.random.randint work differently!
		## random.randint includes the endpoints, numpy does not.
		I = random.randint(0, userids.count()-1)
		logmsg('%i users, selected index %i' % (userids.count(), I))
		uid = userids[I]
		logmsg('selected userid', uid)
		if True:
			if uid is None:
				user = AnonymousUser()
			else:
				user = User.objects.get(id=uid)
			logmsg('Selected user', user)
		qj = QueuedJob.objects.filter(ready=True, priority=toppriority, user__id=uid).order_by('queuedtime')[0]
		qj.ready = False
		qj.save()
		return qj

# need this???
#class RunFailure(models.Model):
#	qjob = models.ForeignKey('QueuedJob', related_name='runs')
#	starttime = models.DateTimeField()
#	endtime = models.DateTimeField()
#	status = models.CharField(max_length=16)
#	failurereason = models.CharField(max_length=256)


