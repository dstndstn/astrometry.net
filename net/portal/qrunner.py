import time
import sys
import os.path

import astrometry.net.ice

## HACK
sys.path.append(os.path.dirname(astrometry.net.ice.__file__))

from astrometry.net.portal.watcher_common import *

from astrometry.net.portal.watcher_script_ice import WatcherIce as watcherclass

from astrometry.net.portal.job import *
from astrometry.net.portal.queue import *
from astrometry.net.portal.log import log as logmsg

while True:
    logmsg('Getting next job...')
    nextjob = QueuedJob.next_job()
    logmsg('Next job is', nextjob)
    if nextjob is None:
        time.sleep(10)
        continue

    logmsg('Running job:', nextjob)
    w = watcherclass()
    w.run_job_or_sub(nextjob.job, nextjob.sub)
    logmsg('Job finished:', nextjob)
    logmsg('  job:', nextjob.job)
    logmsg('  sub:', nextjob.sub)

    requeue = False
    if nextjob.job:
        if nextjob.job.finished_without_error():
            nextjob.delete()
        else:
            requeue = True
    else:
        if nextjob.sub.finished_without_error():
            nextjob.delete()
        else:
            requeue = True

    if requeue:
        logmsg('Job failed: requeuing')
        nextjob.queuedtime = Job.timenow()
        nextjob.priority -= 1
        nextjob.ready = True
        nextjob.save()

