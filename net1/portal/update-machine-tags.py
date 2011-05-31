#! /usr/bin/env python

import astrometry.net.django_commandline

from an.portal.job import Job

if __name__ == '__main__':
    jobs = Job.objects.all()
    NJ = jobs.count()
    for i, job in enumerate(jobs):
        print 'Job %i of %i: %s' % (i, NJ, job.jobid)
        job.remove_all_machine_tags()
        job.add_machine_tags()
        job.save()
