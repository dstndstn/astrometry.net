import os
import sys
# add .. to pythonpath
path = os.path.abspath(__file__)
dotdot = os.path.dirname(os.path.dirname(path))
sys.path.append(dotdot)

# add ../util
sys.path.append(os.path.join(dotdot, 'util'))
from an_build_ext import an_build_ext
from setuputils import *

from distutils.core import setup, Extension

import numpy
numpy_inc = numpy.get_include()

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

c_module = Extension('_plotstuff_c',
                     sources = ['plotstuff_wrap.c'],
                     include_dirs = [numpy_inc] + inc,
                     extra_objects = [
						 'plotstuff.o', 'plotfill.o', 'plotxy.o',
						 'plotimage.o', 'plotannotations.o',
						 'plotgrid.o', 'plotoutline.o', 'plotindex.o',
						 'plotradec.o', 'plothealpix.o', 'plotmatch.o',
						 'matchfile.o', 'matchobj.o',
						 ] + objs,
                         extra_compile_args = cflags,
                         extra_link_args=link,
                         )

setup(cmdclass={'build_ext': an_build_ext},
	  name = 'Plotting stuff in python',
      version = '1.0',
      description = 'Just what you need, another plotting package!',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@astro.princeton.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'plotstuff' ],
      ext_modules = [c_module])

