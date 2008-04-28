import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'qtest.settings'

import sys

from qtest.app.models import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: worker.py <worker-id>'
        sys.exit(-1)
    wid = int(sys.argv[1])
    worker = Worker.objects.get(id=wid)

    while True:
        # select work.
        myinds = [index.id for index in worker.indexes.all()]
        mywork = Work.objects.all().filter(claimed=False, index__id__in=myinds)
        print 'potential work:', mywork

        for work in mywork:
            work.claimed = True
            work.worker = worker
            work.save()

        # perform work...

        for work in mywork:
            fw = FinishedWork(work=work, worker=worker)
            fw.save()

