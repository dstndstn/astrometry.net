from django.db import models

import os
import socket
import thread

from urllib import urlretrieve
import datetime
import time
from datetime import timedelta

from django.core.urlresolvers import reverse

import astrometry.net.settings as settings
import astrometry.net.server.views
from astrometry.net.portal.job import *

class JobQueue(models.Model):
    name = models.CharField(max_length=32)
    queuetype = models.CharField(max_length=32)

    def __str__(self):
        return '%s - %s' % (self.name, self.queuetype)

class QueuedJob(models.Model):
    q = models.ForeignKey(JobQueue, related_name='jobs')
    priority = models.SmallIntegerField(blank=True, default=0)

    #stopwork = models.BooleanField(blank=True, default=False)
    inprogress = models.BooleanField(blank=True, default=False)
    done = models.BooleanField(blank=True, default=False)

    enqueuetime = models.DateTimeField(blank=True, default='2000-01-01')

    job = models.ForeignKey(Job, blank=True, null=True)
    submission = models.ForeignKey(Submission, blank=True, null=True)

    # .work: Work completed so far.
    # .workers: Workers currently working on this job.

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

class Index(models.Model):
    indexid = models.IntegerField()
    healpix = models.IntegerField()
    healpix_nside = models.IntegerField()

class Worker(models.Model):
    hostname = models.CharField(max_length=256, default=socket.gethostname)
    ip = models.IPAddressField(default=lambda: socket.gethostbyname(socket.gethostname()))
    processid = models.IntegerField(default=os.getpid)
    keepalive = models.DateTimeField(blank=True, default=Job.timenow)
    job = models.ForeignKey(QueuedJob, related_name='workers', blank=True, null=True)
    queue = models.ForeignKey(JobQueue, related_name='workers')
    indexes = models.ManyToManyField(Index)

    def __str__(self):
        return self.hostname

    def save(self):
        self.keepalive = datetime.datetime.utcnow()
        super(Worker, self).save()

    def pretty_index_list(self):
        return ', '.join(['%i'%i.indexid + (i.healpix > -1 and '-%02i'%i.healpix or '')
                          for i in self.indexes.all()])

    def start_keepalive_thread(self):
        thread.start_new_thread(Worker.keep_alive, (self.id,))

    @staticmethod
    def filter_keepalive_stale(queryset, allowed_dt):
        cutoff = Worker.get_keepalive_stale_date(allowed_dt)
        return queryset.filter(keepalive__lt=cutoff)

    @staticmethod
    def get_keepalive_stale_date(allowed_dt):
        now = datetime.datetime.utcnow()
        dt = timedelta(seconds=allowed_dt)
        return now - dt

    @staticmethod
    def keep_alive(workerid):
        while True:
            me = Worker.objects.all().get(id=workerid)
            me.save()
            time.sleep(10)


        


class Work(models.Model):
    job = models.ForeignKey(QueuedJob, related_name='work')
    index = models.ForeignKey(Index)
    # could put more fine-grained detail (eg range of field objs) in here.
    worker = models.ForeignKey(Worker, related_name='work', blank=True, null=True)
    inprogress = models.BooleanField(blank=True, default=False)
    done = models.BooleanField(blank=True, default=False)




