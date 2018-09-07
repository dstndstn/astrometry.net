from __future__ import print_function
from distutils.core import setup, Extension
from setuptools.command.install import install
import subprocess
import os
import sys
from glob import glob

class MyInstall(install):
    def run(self):
        print('MyInstall.run: calling "make -k"')
        subprocess.call(['make', '-k'])
        print('MyInstall.run: calling "make -k py"')
        subprocess.call(['make', '-k', 'py'])

        for cmd in ['make -k pyinstall',
                    'make -k install']:
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

class MyBuildExt(install):
    def run(self):
        print('MyBuildExt.run: calling "make -k"')
        subprocess.call(['make', '-k'])
        print('MyBuildExt.run: calling "make -k py"')
        subprocess.call(['make', '-k', 'py'])

setup(name='astrometry',
      version='git',
      author='Astrometry.net team',
      author_email='dstndstn@gmail.com',
      url='http://astrometry.net',
      cmdclass={'install': MyInstall, 'build_ext': MyBuildExt},
      packages=['astrometry'],
      package_dir={'astrometry':''},
      )
