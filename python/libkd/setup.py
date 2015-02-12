from distutils.core import setup, Extension
import numpy
import os.path

from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()

inc = ['../../base', '../../qfits-an', '../../libkd']
libdirs = inc

# for #include libkd/... etc.
inc += ['../..']

# the order is important!
libs = ['kd', 'qfits', 'anbase']

c_module = Extension('libkd._libkd',
                     sources = ['pyspherematch.c'],
                     include_dirs = numpy_inc + inc,
                     library_dirs = libdirs,
                     libraries = libs,
                     )

setup(name = 'Kdtree matching in Python',
      version = '1.0',
      description = 'This package finds near neighbours in two sets of points',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@cmu.edu',
      url = 'http://astrometry.net',
      package_dir={'libkd':''},
      packages=['libkd'],
      ext_modules = [c_module])

