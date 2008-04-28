import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'qtest.settings'

from qtest.app.models import *

if __name__ == '__main__':

    # get Index
    ind1 = Index.objects.get(id=1)

    # set up Work
    for i in xrange(10):
        w = Work(index=ind1)
        w.save()


