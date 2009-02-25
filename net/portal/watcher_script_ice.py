#! /usr/bin/env python

from watcher_common import *
from astrometry.net import settings

from astrometry.net.portal.log import log as logmsg
from astrometry.util.file import *

import IceSolver

class WatcherIce(Watcher):
    def solve_job_and_write_files(self, job):
        blindlog = job.get_filename('blind.log')
        def userlog(msg):
            f = open(blindlog, 'a')
            f.write(msg)
            f.close()

        axy = read_file(job.get_axy_filename())
        logmsg('Calling IceSolver.solve()...')
        files = IceSolver.solve(job.jobid, axy, userlog)
        logmsg('IceSolver.solve() returned')

		if files is not None:
			logmsg('Writing files...')
			basedir = job.get_job_dir()
			for f in files:
				fn = os.path.join(basedir, f.name)
				logmsg('  %i bytes:' % len(f.data), fn)
				write_file(f.data, fn)
		else:
			logmsg('Field unsolved.')
			
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s <input-file>' % sys.argv[0]
        sys.exit(-1)
    joblink = sys.argv[1]
    w = WatcherIce()
    sys.exit(w.main(joblink))

