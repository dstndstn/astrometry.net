from distutils.core import setup, Extension
import numpy
import os.path

numpy_inc = (os.path.dirname(numpy.__file__) +
             '/core/include/numpy')

c_module = Extension('spherematch_c',
                     sources = ['pyspherematch.c'],
                     include_dirs = [ numpy_inc,
                                      '../qfits-an/include',
                                      '../util', '.', ],
#                     extra_objects = ['libkd-noio.a',
                     extra_objects = ['libkd.a',
                                      '../util/libanfiles.a',
                                      '../util/libanutils.a',
                                      '../qfits-an/lib/libqfits.a',
									  ])

setup(name = 'Kdtree matching in Python',
      version = '1.0',
      description = 'This package finds near neighbours in two sets of points',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@cs.toronto.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'spherematch' ],
      ext_modules = [c_module])

