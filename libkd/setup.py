from distutils.core import setup, Extension
import numpy
import os.path

numpy_inc = (os.path.dirname(numpy.__file__) +
             '/core/include/numpy')

module1 = Extension('spherematch',
                    sources = ['pyspherematch.c'],
                    include_dirs = [ numpy_inc ])

setup(name = 'PackageName',
      version = '1.0',
      description = 'This is a demo package',
      ext_modules = [module1])

