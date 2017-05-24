from __future__ import print_function
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from run_command import run_command

def get_libs(pkg, required=True):
    (rtn,out,err) = run_command('pkg-config --libs-only-l ' + pkg)
    if rtn:
        if required:
            raise Exception('Failed to find libraries for package ' + pkg)
        else:
            print('Failed to find libraries for (optional) package', pkg)
            return []
    if err and len(err):
        print('pkg-config complained:', err)
    #print 'pkg-config said:', out
    #libs = out.replace('\n', ' ').split(' ')
    libs = out.split()
    libs = [l for l in libs if len(l)]
    # Strip off the leading "-l"
    libs = [l[2:] for l in libs]
    #print 'returning libs:', libs
    return libs

def get_include_dirs(pkg):
    (rtn,out,err) = run_command('pkg-config --cflags-only-I ' + pkg)
    if rtn:
        raise Exception('Failed to find include paths for package ' + pkg)
    if err and len(err):
        print('pkg-config complained:', err)
    dirs = out.split()
    dirs = [l for l in dirs if len(l)]
    # Strip off the leading "-I"
    dirs = [l[2:] for l in dirs]
    #print 'returning include dirs:', dirs
    return dirs

def get_lib_dirs(pkg, required=True):
    (rtn,out,err) = run_command('pkg-config --libs-only-L ' + pkg)
    if rtn:
        if required:
            raise Exception('Failed to find libraries for package ' + pkg)
        else:
            print('Failed to find libraries for (optional) package', pkg)
            return []
    if err and len(err):
        print('pkg-config said:', err)
    libs = out.split()
    libs = [l for l in libs if len(l)]
    # Strip off the leading "-L"
    libs = [l[2:] for l in libs]
    return libs
