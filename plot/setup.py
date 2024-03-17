from __future__ import print_function
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import os
import sys
# add .. to pythonpath
path = os.path.abspath(__file__)
dotdot = os.path.dirname(os.path.dirname(path))
sys.path.append(dotdot)

# add ../util
sys.path.append(os.path.join(dotdot, 'util'))
from an_build_ext import an_build_ext
from setuputils import *

from setuptools import setup, Extension

import numpy
numpy_inc = numpy.get_include()

def strlist(s, split=' '):
    lst = s.split(split)
    lst = [i.strip() for i in lst]
    lst = [i for i in lst if len(i)]
    return lst

link = ' '.join([os.environ.get('LDFLAGS', ''),
                 os.environ.get('LDLIBS', ''),])
link = strlist(link)
objs = strlist(os.environ.get('SLIB', ''))
inc = strlist(os.environ.get('INC', ''), split='-I')
inc.append('../util') # for util.i
cflags_swig = strlist(os.environ.get('CFLAGS_SWIG', ''))
cflags = strlist(os.environ.get('CFLAGS', ''))

print(('link:', link))
print(('objs:', objs))
print(('inc:', inc))
print(('cflags:', cflags_swig + cflags))

objs = [
    'plotfill.o', 'plotxy.o',
    'plotimage.o', 'plotannotations.o',
    'plotgrid.o', 'plotoutline.o', 'plotindex.o',
    'plotradec.o', 'plothealpix.o', 'plotmatch.o',
    'plotstuff.o', ] + objs

c_module = Extension('_plotstuff_c',
                     sources = [
                         'plotstuff.i',
                         #'plotstuff.c', 'plotfill.c', 'plotxy.c',
                         #'plotimage.c', 'plotannotations.c',
                         #'plotgrid.c', 'plotoutline.c', 'plotindex.c',
                         #'plotradec.c', 'plothealpix.c', 'plotmatch.c',
                         #'matchfile.c', 'matchobj.c',
                         ],
                     include_dirs = [numpy_inc] + inc,
                     depends = objs,
                     extra_objects = objs,
                     extra_compile_args = cflags_swig,
                     extra_link_args=link,
                     swig_opts=['-I'+d for d in inc] + cflags_swig,
    )

setup(cmdclass={'build_ext': an_build_ext},
      name = 'Plotting stuff in python',
      version = '1.0',
      description = 'Just what you need, another plotting package!',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@astro.princeton.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'plotstuff' ],
      ext_modules = [c_module])

