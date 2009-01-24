#! /usr/bin/env python

import astrometry.net.django_commandline

import sys

from astrometry.net.portal.queue import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: qadmin.py <action> [args]'
        print '  <action> can be:'
        print '     "clear" - remove everything from the queue.'
        print '     "show"  - show the queue.'
        sys.exit(-1)

    action = sys.argv[1]
    if action == 'show':
        qjs = QueuedJob.objects.all()
        print '%i queued jobs:' % qjs.count()
        for qj in qjs:
            print '  ', qj

    elif action == 'clear':
        print 'Deleting all queued jobs...'
        QueuedJob.objects.all().delete()
    

