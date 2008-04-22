#! /usr/bin/env python

import sys

from astrometry.net.util.run_command import run_command

if __name__ == '__main__':
    cluster = sys.stdin.readline().strip('\n')

    print 'connecting to', cluster

    cmd = 'ssh -x -T -i ~/.ssh/id_solver_cluster %s' % cluster
    (rtn, out, err) = run_command(cmd)

    if rtn:
        print 'command failed: rtn val %i' % rtn

    print 'out:', out
    print 'err:', err
