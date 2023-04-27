from __future__ import print_function
from distutils.core import setup, Extension
from setuptools.command.install import install
import subprocess
import os
import sys
from glob import glob

version = '0.0'
try:
    v = subprocess.check_output(['git', 'describe'], text=True)
    v = v.strip()
    words = v.split('-')
    if len(words) == 3:
        v = words[0] + '.dev' + words[1]
    version = v
except:
    import traceback
    traceback.print_exc()
    pass

env = dict(AN_GIT_REVISION=version)

class MyInstall(install):
    def run(self):
        print('MyInstall.run: calling "make -k"')
        subprocess.call(['make', '-k'], env=env)
        print('MyInstall.run: calling "make -k py"')
        subprocess.call(['make', '-k', 'py'], env=env)

        for cmd in ['make -k pyinstall',
                    'make -k install']:
            myenv = env.copy()
            dirnm = self.install_base
            if dirnm is not None:
                myenv.update(INSTALL_DIR=dirnm)
            pybase = self.install_platlib
            if pybase is not None:
                pybase = os.path.join(pybase, 'astrometry')
                myenv.update(PY_BASE_INSTALL_DIR=pybase)
            py = sys.executable
            if py is not None:
                myenv.update(PYTHON=py)
            print('Running:', cmd)
            subprocess.call(cmd, shell=True, env=myenv)
            install.run(self)

class MyBuildExt(install):
    def run(self):
        print('MyBuildExt.run: calling "make -k"')
        subprocess.call(['make', '-k'], env=env)
        print('MyBuildExt.run: calling "make -k py"')
        subprocess.call(['make', '-k', 'py'], env=env)

setup(name='astrometry',
      version=version,
      author='Astrometry.net team',
      author_email='dstndstn@gmail.com',
      url='http://astrometry.net',
      cmdclass={'install': MyInstall, 'build_ext': MyBuildExt},
      packages=['astrometry'],
      package_dir={'astrometry':''},
      )
