#! /usr/bin/env python

import astrometry.net.django_commandline

import datetime

from an.portal.job import Job

if __name__ == '__main__':
    jobs = Job.objects.all()
    NJ = jobs.count()

    t0 = datetime.datetime(2000, 1, 1)

    for i, job in enumerate(jobs):
        print 'Job %i of %i: %s' % (i, NJ, job.jobid)
        if job.enqueuetime is None:
            job.set_enqueuetime(t0)
            job.save()
        if job.starttime is None:
            job.set_starttime(t0)
            job.save()
