from distutils.core import setup, Extension
import numpy
import os.path

numpy_inc = (os.path.dirname(numpy.__file__) +
             '/core/include/numpy')

c_module = Extension('spherematch_c',
                     sources = ['pyspherematch.c'],
                     include_dirs = [ numpy_inc,
                                      '../qfits-an/include',
                                      '../util', '.', ],
                     #                     extra_objects = ['libkd-noio.a',
                     extra_objects = ['libkd.a',
                                      '../util/libanfiles.a',
                                      '../util/libanutils.a',
                                      '../qfits-an/lib/libqfits.a',
                                      ])

setup(name = 'pyspherematch',
      ext_package='pyspherematch',
      version = '0.2',
      description = 'This package finds near neighbours in two sets of points. Stand-alone python lib, extracted from Astrometry.net',
      author = 'Astrometry.net (Dustin Lang), python release by Sjoert van Velzen',
      author_email = 'dstn@cs.toronto.edu, s.vanvelzen@astro.ru.nl',
      url = 'http://astrometry.net',
      ext_modules = [c_module],
      packages=['pyspherematch', 'pyspherematch/util'],
      package_dir={'pyspherematch': '.', 'pyspherematch/util': '../util/'})
