

from django.db import models
from django.contrib.auth.models import User

from astrometry.net.portal.job import *


class QueuedJob(models.Model):
    priority_high = 1500
    priority_normal = 1000
    priority_low = 500

    job = models.ForeignKey('Job', related_name='job')
    sub = models.ForeignKey('Submission', related_name='sub')
    # derived from job.user or sub.user for query efficiency.
    user = models.ForeignKey('User', related_name='user')

    queuedtime = models.DateTimeField()
    priority = models.PositiveIntegerField()
    ready = models.BooleanField()

    @staticmethod
    def submit_job(job, priority=priority_normal):
        qj = QueuedJob(job=job.get_user(),
                       user=job.user,
                       queuedtime=Job.timenow(),
                       priority=priority,
                       ready=True)
        qj.save()

    @staticmethod
    def submit_submission(sub, priority=priority_normal):
        qj = QueuedJob(sub=sub,
                       user=sub.user,
                       queuedtime=Job.timenow(),
                       priority=priority,
                       ready=True)
        qj.save()

    #@staticmethod
    #def submit_job_or_submission(js, priority=priority_normal):
    #    if Job.objects.all().get(id=js.id)

    @staticmethod
    def next_job():
        q = QueuedJob.objects.filter(ready=True).values_list('priority', flat=True).order_by('-priority')
        if q.count() == 0:
            return None
        toppriority = q[0]
        #toppriority = q.priority
        #users = QueuedJob.objects.filter(ready=True, priority=toppriority).values_list('job_user', 'sub_user')
        users = QueuedJob.objects.filter(ready=True, priority=toppriority).values_list('user').distinct()
        user = users[random.randint(0, users.count()-1)]
        qj = QueuedJob.objects.filter(ready=True, priority=toppriority, user=user).order_by('queuedtime')[0]
        qj.ready = False
        qj.save()
        return qj

# need this???
class RunFailure(models.Model):
    qjob = models.ForeignKey('QueuedJob', related_name='runs')
    starttime = models.DateTimeField()
    endtime = models.DateTimeField()
    status = models.CharField(max_length=16)
    failurereason = models.CharField(max_length=256)


