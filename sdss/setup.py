from distutils.core import setup, Extension
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))
from an_build_ext import an_build_ext

setup(cmdclass={'build_ext': an_build_ext},
	  name='C utils for Astrometry.net SDSS routines',
	  version = '1.0',
	  author = 'Lang & Hogg',
	  author_email = 'dstn@cmu.edu',
	  url = 'http://astrometry.net',
	  ext_modules = [Extension('_cutils', ['cutils.i'], libraries=['m'])],
	)
