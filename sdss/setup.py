from distutils.core import setup, Extension

setup(name='C utils for Astrometry.net SDSS routines',
	  version = '1.0',
	  author = 'Lang & Hogg',
	  author_email = 'dstn@astro.princeton.edu',
	  url = 'http://astrometry.net',
	  py_modules = ['cutils'],
	  ext_modules = [Extension('_cutils', ['cutils.i'])],
	  #swig_opts = ['-modern'],
	  )
