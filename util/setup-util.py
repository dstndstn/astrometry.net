import os
import sys
# add .. to pythonpath
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(path)))

from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
from setuputils import *

numpy_inc = get_numpy_include_dirs()

c_swig_module = Extension('_util',
						  sources = ['util_wrap.c' ],
						  include_dirs = numpy_inc +
						  [
							  '../qfits-an/include',
							  '../libkd',
							  '.'],
						  extra_objects = [
							  'libanfiles.a',
							  '../libkd/libkd.a',
							  'libanutils.a',
							  '../qfits-an/lib/libqfits.a',
							  '../gsl-an/libgsl-an.a',
							  ],
						  extra_link_args=[os.environ.get('WCSLIB_LIB', ''),
#os.environ.get('GSL_LIB', ''),
#'-O0 -g'
										   ],
						  #extra_compile_args = ['-O0 -g'],

						  )

setup(name = 'Access to Astrometry.net utils in python',
      version = '1.0',
      description = '',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@astro.princeton.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'util' ],
	  ext_modules = [c_swig_module])

