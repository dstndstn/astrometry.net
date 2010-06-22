from distutils.core import setup, Extension

# HACK -- get numpy include path.
import numpy
import os.path
numpy_inc = (os.path.dirname(numpy.__file__) +
             '/core/include/numpy')


# This could also be done via swig's %native()...
c_util_module = Extension('_index_util',
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

c_swig_module = Extension('_index_c',
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

setup(name = 'Access to Astrometry.net index files in python',
      version = '1.0',
      description = '',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@astro.princeton.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'index' ],
	  ext_modules = [c_swig_module, c_util_module])

