from __future__ import print_function
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
import os
import sys
# add .. to pythonpath
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from an_build_ext import *

from setuptools import setup, Extension
import numpy
from setuputils import *

numpy_inc = [numpy.get_include()]

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
cflags = strlist(os.environ.get('CFLAGS', ''))

print('link:', link)
print('objs:', objs)
print('inc:', inc)
print('cflags:', cflags)

c_swig_module = Extension('_util',
                          sources = ['util.i'],
                          include_dirs = numpy_inc + inc + ['.'],
                          extra_objects = objs,
                          extra_compile_args = cflags,
                          extra_link_args=link,
                          depends=objs,
                          swig_opts=['-I'+d for d in inc],
    )

setup(cmdclass={'build_ext': an_build_ext},
      name = 'Access to Astrometry.net utils in python',
      version = '1.0',
      description = '',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstndstn@gmail.com',
      url = 'http://astrometry.net',
      py_modules = [ 'util' ],
      ext_modules = [c_swig_module])

