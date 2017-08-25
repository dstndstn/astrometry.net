from __future__ import print_function
from distutils.core import setup, Extension
from setuptools.command.install import install
import subprocess
import os
import sys
from glob import glob

# from https://stackoverflow.com/questions/33168482/compiling-installing-c-executable-using-pythons-setuptools-setup-py
# def get_virtualenv_path():
#     if hasattr(sys, 'real_prefix'):
#         return sys.prefix
#     if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
#         return sys.prefix
#     if 'conda' in sys.prefix:
#         return sys.prefix
#     return None

class MyInstall(install):
    def run(self):
        print('MyInstall.run: calling "make"')
        subprocess.call(['make', '-k'])
        print('MyInstall.run: calling "make py"')
        subprocess.call(['make', '-k', 'py'])

        cmd = 'make -k install'
        dirnm = self.install_base
        if dirnm is not None:
            cmd += ' INSTALL_DIR="%s"' % dirnm
        pybase = self.install_platlib
        if pybase is not None:
            pybase = os.path.join(pybase, 'astrometry')
            cmd += ' PY_BASE_INSTALL_DIR="%s"' % pybase
        py = sys.executable
        if py is not None:
            cmd += ' PYTHON="%s"' % py
        print('Running:', cmd)
        subprocess.call(cmd, shell=True)
        install.run(self)

# util_execs = [
#     'an-fitstopnm', 'an-pnmtofits', 'casjobs.py', 'downsample-fits',
#     'fit-wcs', 'fits-column-merge', 'fits-flip-endian', 'fitsgetext',
#     'get-healpix', 'hpsplit', 'query-starkd', 'subtable', 'tabsort',
#     'wcs-match', 'wcs-pv2sip', 'wcs-rd2xy', 'wcs-resample',
#     'wcs-to-tan', 'wcs-xy2rd', 'wcsinfo',
# ]
# catalog_execs = [
#     '2masstofits', 'build-hd-tree', 'nomadtofits', 'tycho2tofits', 'usnobtofits'
#     ]
# blind_execs = [
#     'astrometry-engine', 'augment-xylist', 'build-astrometry-index', 'fits-guess-scale',
#     'fitscopy', 'fitsverify', 'image2xy', 'imarith', 'imcopy', 'imstat', 'listhead',
#     'liststruc', 'modhead', 'new-wcs', 'plotann.py', 'solve-field', 'tablist', 'tabmerge',
#     'wcs-grab'
#     ]
    
setup(name='astrometry',
      version='git',
      author='Astrometry.net team',
      author_email='dstndstn@gmail.com',
      url='http://astrometry.net',
      cmdclass={'install': MyInstall},
      packages=['astrometry'],
      package_dir={'astrometry':''},
      # data_files=[
      #     ('bin', (
      #         [os.path.join('util', p)     for p in util_execs   ] +
      #         [os.path.join('catalogs', p) for p in catalog_execs] +
      #         [os.path.join('blind', p)    for p in blind_execs  ]
      #      )),
      #      ('etc', [os.path.join('etc', 'astrometry.cfg')]),
      #      (os.path.join('include','astrometry'), glob(os.path.join('include', 'astrometry', '*'))),
      #     ],
      )

# fixme -- os-features-config.h ?
#    -- bin/*
#    -- doc/* ?
