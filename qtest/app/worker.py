import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'qtest.settings'

import sys
import time

from qtest.app.models import *

from django.db import connection
from django.db import transaction

@transaction.commit_manually
def grab_work(worker):
    cursor = connection.cursor()
    #print 'Locking table...'
    cursor.execute('LOCK TABLE app_work IN SHARE ROW EXCLUSIVE MODE')
    #print 'Locked table.'

    # select work.
    myinds = [index.id for index in worker.indexes.all()]
    mywork = Work.objects.all().filter(claimed=False, index__id__in=myinds)
    #print 'potential work:', mywork

    # artificially long delay...
    #time.sleep(1)

    for work in mywork:
        print 'Claiming work', work
        work.claimed = True
        work.worker = worker
        work.save()

    transaction.commit()
    print '.',
    sys.stdout.flush()
    return mywork

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: worker.py <worker-id>'
        sys.exit(-1)
    wid = int(sys.argv[1])
    worker = Worker.objects.get(id=wid)

    while True:

        print 'Looking for work...'
        while True:
            time.sleep(1)
            mywork = grab_work(worker)
            if len(mywork) == 0:
                continue
            break
        print
        # perform work...
        for w in mywork:
            print 'Doing work on', 

        for work in mywork:
            print 'Finished work', w
            fw = FinishedWork(work=work, worker=worker)
            fw.save()
            print 'Added FinishedWork', fw.id

