# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import

import os
try:
    # py2
    import cPickle as pickle
except:
    import pickle

def trymakedirs(fn, dir=False):
    if dir is True:
        dirnm = os.path.dirname(fn)
    else:
        dirnm = fn
    if not os.path.exists(dirnm):
        try:
            os.makedirs(dirnm)
        except:
            pass

def _filesize(fn):
    st = os.stat(fn)
    return st.st_size

def file_size(fn, error=None):
    if error is None:
        return _filesize(fn)
    try:
        return _filesize(fn)
    except OSError:
        return error

def read_file(fn):
    return open(fn).read()

def write_file(data, fn):
    f = open(fn, 'wb')
    f.write(data)
    f.close()
    
def pickle_to_file(data, fn):
    f = open(fn, 'wb')
    # MAGIC -1: highest pickle protocol
    pickle.dump(data, f, -1)
    f.close()

def unpickle_from_file(fn):
    f = open(fn, 'rb')
    data = pickle.load(f)
    # necessary?
    f.close()
    return data

def get_svn_version():
    from .run_command import run_command
    version = {}
    rtn,out,err = run_command('svn info')
    if rtn != 0:
        import sys
        print('Error getting SVN version: rtn', rtn, '\nOut:', out, '\nErr:', err, file=sys.stderr)
    assert(rtn == 0)
    lines = out.split('\n')
    lines = [l for l in lines if len(l)]
    for l in lines:
        words = l.split(':', 1)
        words = [w.strip() for w in words]
        version[words[0]] = words[1]
    return version

def get_git_version():
    '''
    eg,
    {'commit': 'a5c7865efd188715a8436ef7be23e38448e2aa60', 'describe': 'v1.0'}
    '''
    from .run_command import run_command
    version = {}
    rtn,out,err = run_command('git log --max-count=1 | head -n 1')
    assert(rtn == 0)
    lines = out.split('\n')
    lines = [l for l in lines if len(l)]
    for l in lines:
        words = l.split(' ', 1)
        words = [w.strip() for w in words]
        version[words[0]] = words[1]

    rtn,out,err = run_command('git describe')
    if rtn == 0:
        # this can fail if there has been no "git tag"
        lines = out.split('\n')
        lines = [l for l in lines if len(l)]
        assert(len(lines) == 1)
        version['describe'] = lines[0]

    return version

