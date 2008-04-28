import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'qtest.settings'

from qtest.app.models import *

if __name__ == '__main__':

    # set up Indexes
    Index.objects.all().delete()

    ind1 = Index(id=1)
    ind1.save()

    ind2 = Index(id=2)
    ind2.save()

    # set up Workers
    Worker.objects.all().delete()

    w1 = Worker(id=1)
    w1.save()
    w1.indexes.add(ind1)
    w1.indexes.add(ind2)

    w2 = Worker(id=2)
    w2.save()
    w2.indexes.add(ind1)
    w2.indexes.add(ind2)

    # set up Work
    Work.objects.all().delete()

    for i in xrange(10):
        w = Work(index=ind1)
        w.save()


