from django.db import models

from urllib import urlretrieve
from datetime import datetime, timedelta

from django.core.urlresolvers import reverse

import views
import settings

class JobQueue(models.Model):
    name = models.CharField(max_length=32, unique=True, primary_key=True)

class QueuedJob(models.Model):
    q = models.ForeignKey(JobQueue, related_name='jobs')
    priority = models.SmallIntegerField(blank=True, default=0)
    jobid = models.CharField(max_length=32)
    stopwork = models.BooleanField(blank=True, default=False)
    enqueuetime = models.DateTimeField(blank=True, default='2000-01-01')
    # .work: Work completed so far.
    # .workers: Workers current working on this job.

    def __str__(self):
        return 'QueuedJob: %s' % self.jobid

    def get_url(self):
        return (settings.MAIN_SERVER + reverse(views.get_input) + '?jobid=%s' % self.jobid)

    def get_put_results_url(self):
        return (settings.MAIN_SERVER + reverse(views.set_results) + '?jobid=%s' % self.jobid)

    def retrieve_to_file(self, fn=None):
        if fn is not None:
            (fn, hdrs) = urlretrieve(self.get_url(), fn)
            return fn
        else:
            (fn, hdrs) = urlretrieve(self.get_url())
            return fn

class Worker(models.Model):
    hostname = models.CharField(max_length=256)
    ip = models.IPAddressField()
    processid = models.IntegerField()
    job = models.ForeignKey(QueuedJob, related_name='workers', blank=True, null=True)
    keepalive = models.DateTimeField(blank=True, default='2000-01-01')

    def __str__(self):
        return self.hostname

    def save(self):
        self.keepalive = datetime.utcnow()
        super(Worker, self).save()

    def pretty_index_list(self):
        return ', '.join(['%i'%i.indexid + (i.healpix > -1 and '-%02i'%i.healpix or '')
                          for i in self.indexes.all()])

    @staticmethod
    def filter_keepalive_stale(queryset, allowed_dt):
        cutoff = Worker.get_keepalive_stale_date(allowed_dt)
        return queryset.filter(keepalive__lt=cutoff)

    @staticmethod
    def get_keepalive_stale_date(allowed_dt):
        now = datetime.utcnow()
        dt = timedelta(seconds=allowed_dt)
        return now - dt

class Index(models.Model):
    indexid = models.IntegerField()
    healpix = models.IntegerField()
    healpix_nside = models.IntegerField()
    # this is probably not required... might help for management, though.
    worker = models.ForeignKey(Worker, related_name='indexes')

class Work(models.Model):
    job = models.ForeignKey(QueuedJob, related_name='work')
    #index = models.ForeignKey(Index)
    worker = models.ForeignKey(Worker, related_name='work')
    inprogress = models.BooleanField(blank=True, default=False)
    done = models.BooleanField(blank=True, default=False)




