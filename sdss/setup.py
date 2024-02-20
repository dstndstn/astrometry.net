# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from setuptools import setup, Extension
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'util'))
from an_build_ext import an_build_ext

setup(cmdclass={'build_ext': an_build_ext},
      name='C utils for Astrometry.net SDSS routines',
      version = '1.0',
      author = 'Lang & Hogg',
      author_email = 'dstndstn@gmail.com',
      url = 'http://astrometry.net',
      ext_modules = [Extension('_cutils', ['cutils.i'], libraries=['m'])],
    )
