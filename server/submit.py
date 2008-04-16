#! /usr/bin/env python

import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.server.settings'

import time
import sys
from datetime import datetime

import astrometry.server.settings as settings
from astrometry.server.models import *

#from astrometry.net.portal.job import Job

def now():
    return datetime.utcnow()

def main(jobqueue, jobid, axyfile):
    print 'Queue', jobqueue
    print 'Jobid', jobid
    print 'Axyfile', axyfile

    q = JobQueue.objects.get(name=jobqueue)

    job = QueuedJob(q=q, jobid=jobid, enqueuetime=now())
    job.save()

    for i in range(15):
        jobs = QueuedJob.objects.all().filter(jobid=jobid)
        if not len(jobs):
            break
        job = jobs[0]
        if job.stopwork:
            break
        print 'Sleeping... %i workers' % job.workers.count()
        sys.stdout.flush()
        time.sleep(3)


    print 'Quitting.'
    sys.stdout.flush()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print >> sys.stderr, 'Usage: submit.py <jobqueue> <jobid> <axyfile>'
        sys.exit(-1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))

