import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

import time
import datetime

import astrometry.net.settings as settings
from astrometry.net.server.models import *

def clean_db():
    print 'Cleaning db...'

    late = Worker.objects.all()
    late = Worker.filter_keepalive_stale(late, 30)
    if len(late):
        print 'Deleting workers who have not stamped their keepalives:'
        stale = Worker.get_keepalive_stale_date(30)
        print 'Now is    ', datetime.datetime.utcnow()
        print 'Cutoff was', stale
        for w in late:
            print '  %s: timestamp %s, missed by %s' % (str(w), str(w.keepalive), str(stale - w.keepalive))
        late.delete()

    donejobs = QueuedJob.objects.all().filter(done=True)
    for job in donejobs:
        if job.workers.all().count():
            continue
        print 'Removing finished job from queue:', job
        job.delete()



if __name__ == '__main__':
    while True:
        clean_db()
        time.sleep(10)
        
