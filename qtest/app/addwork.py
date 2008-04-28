import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'qtest.settings'
import time

from qtest.app.models import *

if __name__ == '__main__':

    Work.objects.all().delete()

    # get Index
    ind1 = Index.objects.get(id=1)

    # set up Work
    ids = []
    for i in xrange(10):
        w = Work(index=ind1)
        w.save()
        ids.append(w.id)

    print 'Added work - waiting for it to be finished.'

    while True:
        fw = FinishedWork.objects.all().filter(work__id__in=ids)
        print '%i of %i done' % (fw.count(), len(ids))
        print 'Ids:', ids
        print 'All work:'
        for w in Work.objects.all():
            print '  ', w
        time.sleep(1)

