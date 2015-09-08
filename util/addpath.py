# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import sys
import os

def addpath():
    try:
        import astrometry
        from astrometry.util.shell import shell_escape
        from astrometry.util.filetype import filetype_short
    except ImportError:
        me = __file__
        path = os.path.realpath(me)
        utildir = os.path.dirname(path)
        assert(os.path.basename(utildir) == 'util')
        andir = os.path.dirname(utildir)
        if os.path.basename(andir) == 'astrometry':
            rootdir = os.path.dirname(andir)
            sys.path.insert(1, andir)
        else:
            # assume there's a symlink astrometry -> .
            rootdir = andir
        #sys.path += [rootdir]
        sys.path.insert(1, rootdir)
