#! /usr/bin/env python

from watcher_common import *
from astrometry.net.server import ssh_master

class WatcherDsolver(Watcher):
    def solve_job(self, job):
        def logfunc(s):
            f = open(job.get_filename('blind.log'), 'a')
            f.write(s)
            f.close()
        tardata = ssh_master.solve(job, logfunc)
        return tardata


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s <input-file>' % sys.argv[0]
        sys.exit(-1)
    joblink = sys.argv[1]
    w = WatcherDsolver()
    sys.exit(w.main(joblink))

