from django.db import models

class Index(models.Model):
    def __str__(self):
        return 'Index-%i' % self.id

class Worker(models.Model):
    indexes = models.ManyToManyField(Index, related_name='workers')
    def __str__(self):
        return 'Worker-%i' % self.id

class Work(models.Model):
    index = models.ForeignKey(Index)
    claimed = models.BooleanField(blank=True, default=False)
    worker = models.ForeignKey(Worker, blank=True, default=None, null=True)

    def __str__(self):
        return 'Work-%i:(%s,%s,%s)' % (self.id, str(self.index),
                                       str(self.worker),
                                       self.claimed and 'claimed' or 'unclaimed')


class FinishedWork(models.Model):
    work = models.ForeignKey(Work)
    worker = models.ForeignKey(Worker)

    def __str__(self):
        return 'FinishedWork-%i:(%s,%s,%s)' % (self.id, 
                                               str(self.work),
                                               str(self.worker))
