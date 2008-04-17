import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.server.settings'

import time
from datetime import datetime
from datetime import timedelta

import astrometry.server.settings as settings
from astrometry.server.models import *

def clean_db():
    print 'Cleaning db...'
    now = datetime.utcnow()
    # Worker.keepalive should be updated every 10 seconds;
    # we give it 30.
    dt = timedelta(seconds=30)
    cutoff = now - dt

    late = Worker.objects.all().filter(keepalive__lt=cutoff)
    if len(late):
        print 'Deleting workers who have not stamped their keepalives:', late
        late.delete()

    donejobs = QueuedJob.objects.all().filter(stopwork=True)
    for job in donejobs:
        if job.workers.all().count():
            continue
        print 'Removing finished job from queue:', job
        job.delete()

    time.sleep(10)


if __name__ == '__main__':
    while True:
        clean_db()
        
