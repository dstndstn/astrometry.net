from distutils.core import setup, Extension
import numpy
import os.path

from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()
print 'Numpy inc:', numpy_inc

def strlist(s, split=' '):
    lst = s.split(split)
    lst = [i.strip() for i in lst]
    lst = [i for i in lst if len(i)]
    return lst

link = ' '.join([os.environ.get('LDFLAGS', ''),
                 os.environ.get('LDLIBS', ''),])
link = strlist(link)
#objs = strlist(os.environ.get('SLIB', ''))
inc = strlist(os.environ.get('INC', ''), split='-I')
cflags = strlist(os.environ.get('CFLAGS', ''))

print 'link:', link
#print 'objs:', objs
print 'inc:', inc
print 'cflags:', cflags

c_module = Extension('spherematch_c',
                     sources = ['pyspherematch.c'],
                     include_dirs = numpy_inc + inc,
                     extra_objects = ['libkd-min.a'],
                     extra_compile_args = cflags,
                     extra_link_args=link,
    )

setup(name = 'Kdtree matching in Python',
      version = '1.0',
      description = 'This package finds near neighbours in two sets of points',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@cmu.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'spherematch' ],
      ext_modules = [c_module])

