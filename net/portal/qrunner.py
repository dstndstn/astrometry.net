import os
import time

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


