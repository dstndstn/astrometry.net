from distutils.core import setup, Extension

c_module = Extension('_index_c',
                     sources = ['index_wrap.c'],
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
	  ext_modules = [c_module])

