from distutils.core import setup, Extension

import numpy
import os.path
from an_build_ext import *
from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()
# print 'numpy_inc:', numpy_inc
numpy_inc = ' '.join([os.path.join(x, 'numpy') for x in numpy_inc])

# This could also be done via swig's %native()...
c_util_module = Extension(
	'_index_util',
	sources = ['index_pyutils.c' ],
	include_dirs = [
		numpy_inc,
		'../qfits-an/include',
		'../libkd',
		'.'],
	extra_objects = [
		'libanfiles.a',
		'../libkd/libkd.a',
		'libanutils.a',
		'../qfits-an/lib/libqfits.a',
		],
	extra_compile_args=['-O0','-g'],
	extra_link_args=['-O0', '-g'],
	)

c_swig_module = Extension(
	'_index_c',
	sources = ['index_wrap.c' ],
	include_dirs = [
		'../qfits-an/include',
		'../libkd',
		'.'],
	extra_objects = [
		'libanfiles.a',
		'../libkd/libkd.a',
		'libanutils.a',
		'../qfits-an/lib/libqfits.a',
		],
	extra_compile_args=['-O0','-g'],
	extra_link_args=['-O0', '-g'],
	)

setup(cmdclass={'build_ext': an_build_ext},
	  name = 'Access to Astrometry.net index files in python',
      version = '1.0',
      description = '',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@cmu.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'index' ],
	  ext_modules = [c_swig_module, c_util_module])

