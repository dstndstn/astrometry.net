from distutils.core import setup, Extension
from setuptools.command.install import install
import subprocess
import sys

# from https://stackoverflow.com/questions/33168482/compiling-installing-c-executable-using-pythons-setuptools-setup-py
def get_virtualenv_path():
    if hasattr(sys, 'real_prefix'):
        return sys.prefix
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return sys.prefix
    if 'conda' in sys.prefix:
        return sys.prefix
    return None

class MyInstall(install):
    def run(self):
        print('MyInstall.run: calling "make"')
        subprocess.call('make')
        print('MyInstall.run: calling "make py"')
        subprocess.call(['make', 'py'])
        cmd = 'make install'
        install_dir = get_virtualenv_path()
        print('Install dir:', install_dir)
        if install_dir is not None:
            cmd += ' INSTALL_DIR="%s"' % install_dir
        print('Running:', cmd)
        subprocess.call(cmd, shell=True)
        install.run(self)
        
setup(name='astrometry',
      version='git',
      author='Astrometry.net team',
      author_email='dstndstn@gmail.com',
      url='http://astrometry.net',
      cmdclass={'install': MyInstall},
      packages=['astrometry'],
      package_dir={'astrometry':''},
      )
#data_files=[
#          ('bin', ['util/

