from django.db import models

from urllib import urlretrieve

class JobQueue(models.Model):
    name = models.CharField(max_length=32, unique=True, primary_key=True)

class QueuedJob(models.Model):
    q = models.ForeignKey(JobQueue, related_name='jobs')
    priority = models.SmallIntegerField(blank=True, default=0)
    jobid = models.CharField(max_length=32)
    stopwork = models.BooleanField(blank=True, default=False)
    enqueuetime = models.DateTimeField(blank=True, default='2000-01-01')
    # .work: Work completed so far.

    #axyurl = models.CharField(max_length=1024)

    def __str__(self):
        return 'QueuedJob: %s' % self.jobid

    def get_url(self):
        # HACK
        return 'http://oven.cosmo.fas.nyu.edu:8888/server/input/?jobid=%s' % self.jobid

    def get_put_results_url(self):
        # HACK
        return 'http://oven.cosmo.fas.nyu.edu:8888/server/results/?jobid=%s' % self.jobid

    #def get_file(self):

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
    job = models.ForeignKey(QueuedJob, related_name='workers', blank=True, null=True)

    def pretty_index_list(self):
        return ', '.join(['%i'%i.indexid + (i.healpix > -1 and '-%i'%i.healpix or '')
                          for i in self.indexes.all()])



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

#class LoadedIndex(models.Model):
#index = models.ForeignKey(Worker, related_name='indexes')



