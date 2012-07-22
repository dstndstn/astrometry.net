from distutils.core import setup, Extension

setup(name='C utils for Astrometry.net SDSS routines',
	  version = '1.0',
	  author = 'Lang & Hogg',
	  author_email = 'dstn@astro.princeton.edu',
	  url = 'http://astrometry.net',
	  ext_modules = [Extension('_cutils', ['cutils.i'])],
	  #py_modules = ['cutils'],
	  #swig_opts = ['-modern'],
	  )
