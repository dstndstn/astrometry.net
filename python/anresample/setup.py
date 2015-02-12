import os
import glob
from setuptools import setup, Extension

from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()

# TODO this extension doesn't actually wrap the resample C code, right?
inc = []
libs = []
srcs = ['_resample.i']

pkg = 'anresample'

ext = Extension(pkg + '.__resample',
                sources = srcs,
                include_dirs = numpy_inc + inc,
                library_dirs = inc,
                libraries = libs,
                )

setup(name=pkg,
      version='1.0',
      author='Dustin Lang, Keir Mierle',
      author_email='dstndstn@gmail.com',
      url='http://astrometry.net',
      ext_modules=[ext],
      packages=[pkg],
      package_dir={pkg:''},
    )

