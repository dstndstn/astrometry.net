#! /usr/bin/env python

import os
import sys

from astrometry.util.run_command import run_command

if __name__ == '__main__':
    print 'Waiting for cluster machine name on stdin...'
    sys.stdout.flush()
    cluster = sys.stdin.readline().strip('\n')

    print 'connecting to', cluster
    sys.stdout.flush()

    cmd = 'ssh -x -T -i ~/.ssh/id_solver_cluster %s' % cluster
    print 'command', cmd
    sys.stdout.flush()
    #(rtn, out, err) = run_command(cmd)
    #if rtn:
    #    print 'command failed: rtn val %i' % rtn
    #print 'out:', out
    #print 'err:', err
    print 'command:', cmd

    os.system(cmd)
    
