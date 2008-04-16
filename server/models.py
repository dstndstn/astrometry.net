from django.db import models

class JobQueue(models.Model):
    name = models.CharField(max_length=32, unique=True, primary_key=True)

class QueuedJob(models.Model):
    q = models.ForeignKey(JobQueue, related_name='jobs')
    priority = models.SmallIntegerField(blank=True, default=0)
    jobid = models.CharField(max_length=32)
    stopwork = models.BooleanField(blank=True, default=False)

class Worker(models.Model):
    hostname = models.CharField(max_length=256)
    ip = models.IPAddressField()
    job = models.ForeignKey(QueuedJob, related_name='workers')

class LoadedIndex(models.Model):
    indexid = models.IntegerField()
    healpix = models.IntegerField()
    healpix_nside = models.IntegerField()
    worker = models.ForeignKey(Worker, related_name='indexes')



