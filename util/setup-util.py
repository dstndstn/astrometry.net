from distutils.core import setup, Extension

# HACK -- get numpy include path.
#import numpy
#import os.path
#numpy_inc = (os.path.dirname(numpy.__file__) +
#             '/core/include/numpy')


c_swig_module = Extension('_util',
                     sources = ['util_wrap.c' ],
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

setup(name = 'Access to Astrometry.net utils in python',
      version = '1.0',
      description = '',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@astro.princeton.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'util' ],
	  ext_modules = [c_swig_module])

