import os
import sys
# add .. to pythonpath
path = os.path.abspath(__file__)
dotdot = os.path.dirname(os.path.dirname(path))
sys.path.append(dotdot)
#sys.path.append(os.path.dirname(dotdot))
#print 'sys.path is', sys.path
from distutils.core import setup, Extension
#from astrometry.util.setuputils import *
from util.setuputils import *

import numpy
numpy_inc = numpy.get_include()

netpbm_inc = os.environ.get('NETPBM_INC', '')
netpbm_lib = os.environ.get('NETPBM_LIB', '-lnetpbm')

jpeg_inc = os.environ.get('JPEG_INC', '')
jpeg_lib = os.environ.get('JPEG_LIB', '-ljpeg')

extra_inc_dirs = []
compile_args = []

# Pull "-I/dir" into extra_inc_dirs
for w in (' '.join([netpbm_inc, jpeg_inc])).split(' '):
	print 'word "%s"' % w
	if len(w) == 0:
		continue
	if w.startswith('-I'):
		extra_inc_dirs.append(w[2:])
	else:
		compile_args.append(w)

extra_link_dirs = []
extra_link_libs = []
link_args = []

for w in (' '.join([netpbm_lib, jpeg_lib])).split(' '):
	print 'word "%s"' % w
	if len(w) == 0:
		continue
	if w.startswith('-L'):
		extra_link_dirs.append(w[2:])
	elif w.startswith('-l'):
		extra_link_libs.append(w[2:])
	else:
		link_args.append(w)

c_module = Extension('_plotstuff_c',
                     sources = ['plotstuff_wrap.c'],
                     include_dirs = ([
						 '../qfits-an/include',
						 '../libkd',
						 '../util', '.'] +
									 get_include_dirs('cairo') +
									 [numpy_inc] +
									 extra_inc_dirs),
					 extra_objects = [
						 'plotstuff.o', 'plotfill.o', 'plotxy.o',
						 'plotimage.o', 'plotannotations.o',
						 'plotgrid.o', 'plotoutline.o', 'plotindex.o',
						 'plotradec.o', 'plothealpix.o', 'plotmatch.o',
						 'matchfile.o', 'matchobj.o',
						 '../catalogs/libcatalogs.a',
						'../util/cairoutils.o',
						 '../util/libanfiles.a',
						 '../libkd/libkd.a',
						 '../util/libanutils.a',
						 '../qfits-an/lib/libqfits.a',
						 '../gsl-an/libgsl-an.a',
						 ],
					 libraries=reduce(lambda x,y: x+y, [get_libs(x,req) for x,req in [('cairo',True), ('wcslib',False)]]) + ['jpeg'] + extra_link_libs,
					 library_dirs=reduce(lambda x,y: x+y, [get_lib_dirs(x,req) for x,req in [('cairo',True), ('wcslib',False)]]) + extra_link_dirs,
		     extra_compile_args=compile_args,
		     extra_link_args=link_args,
					 )

setup(name = 'Plotting stuff in python',
      version = '1.0',
      description = 'Just what you need, another plotting package!',
      author = 'Astrometry.net (Dustin Lang)',
      author_email = 'dstn@astro.princeton.edu',
      url = 'http://astrometry.net',
      py_modules = [ 'plotstuff' ],
      ext_modules = [c_module])

