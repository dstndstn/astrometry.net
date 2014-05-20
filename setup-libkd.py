import os
from distutils.core import setup, Extension

from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()

inc = ['util', 'qfits-an']

libkd_srcs = [
    'pyspherematch.c',
    'dualtree.c', 'dualtree_rangesearch.c', 'dualtree_nearestneighbour.c',
    'kdtree.c', 'kdtree_dim.c', 'kdtree_mem.c',
    'kdtree_fits_io.c',
    'kdint_ddd.c',
    'kdint_fff.c',
    'kdint_ddu.c',
    'kdint_duu.c',
    'kdint_dds.c',
    'kdint_dss.c',
    ]
util_srcs = [
    'ioutils.c', 'bl.c', 'mathutil.c', 'fitsioutils.c', 'fitsbin.c',
    'an-endian.c', 'fitsfile.c', 'log.c', 'errors.c', 'tic.c',
    ]

qfits_srcs = [
    'anqfits.c', 'qfits_tools.c', 'qfits_table.c', 'qfits_float.c',
    'qfits_error.c', 'qfits_time.c', 'qfits_card.c', 'qfits_header.c',
    'qfits_rw.c', 'qfits_memory.c', 'qfits_convert.c', 'qfits_byteswap.c',
    ]

srcs = ([os.path.join('libkd',x) for x in libkd_srcs] +
        [os.path.join('util', x) for x in util_srcs]  +
        [os.path.join('qfits-an', x) for x in qfits_srcs]
    )

    
ext = Extension('astrometry.libkd.spherematch_c',
                sources = srcs,
                include_dirs = numpy_inc + inc,
                )

setup(name='libkd',
      version='1.0',
      author='Dustin Lang, Keir Mierle',
      author_email='dstndstn@gmail.com',
      url='http://astrometry.net',
      ext_modules=[ext],
      py_modules=['astrometry.libkd.spherematch',
                  'astrometry.util.starutil_numpy'],
      package_dir={'astrometry':''},
      data_files=[('tests', ['libkd/test_spherematch.py']),],
    )

