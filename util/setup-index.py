from distutils.core import setup, Extension

import numpy
import os.path
from an_build_ext import *
from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()
# print 'numpy_inc:', numpy_inc
numpy_inc = ' '.join([os.path.join(x, 'numpy') for x in numpy_inc])

def strlist(s, split=' '):
    lst = s.split(split)
    lst = [i.strip() for i in lst]
    lst = [i for i in lst if len(i)]
    return lst

link = ' '.join([os.environ.get('LDFLAGS', ''),
                 os.environ.get('LDLIBS', ''),])
link = strlist(link)
objs = strlist(os.environ.get('SLIB', ''))
inc = strlist(os.environ.get('INC', ''), split='-I')
cflags = strlist(os.environ.get('CFLAGS', ''))

print 'link:', link
print 'objs:', objs
print 'inc:', inc
print 'cflags:', cflags

c_swig_module = Extension('_index_util',
						  sources = ['index_pyutils.c'],
						  include_dirs = numpy_inc + inc + ['.'],
						  extra_objects = objs,
                          extra_compile_args = cflags,
                          extra_link_args=link,
	)
c_swig_module = Extension('_index_c',
						  sources = ['index_wrap.c'],
						  include_dirs = numpy_inc + inc + ['.'],
						  extra_objects = objs,
                          extra_compile_args = cflags,
                          extra_link_args=link,
	)

setup(cmdclass={'build_ext': an_build_ext},
	  name = 'Access to Astrometry.net index files in python',
      version = '1.0',
      description = '',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@cmu.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'index' ],
	  ext_modules = [c_swig_module, c_util_module])

