from distutils.core import setup, Extension

LEVMAR_LIBDIR = '/usr/local/lib'
LEVMAR_INCDIR = '/usr/local/include'

levmar = Extension('levmar', ['src/pylevmar.c'],
                   libraries = ['levmar', 'm', 'blas', 'lapack'],
                   extra_compile_args = ['-g'],
                   library_dirs = ['/usr/lib', LEVMAR_LIBDIR],
                   include_dirs = [LEVMAR_INCDIR],
                   depends = ['src/pylevmar.h'])

setup(name = 'pylevmar',
      version = '0.1',
      description = 'Python Bindings to levmar',
      author = 'Alastair Tse',
      author_email = 'alastair@liquidx.net',
      url = 'http://www.liquidx.net/pylevmar/',
      license = 'BSD',
      ext_modules = [levmar]
      )
                                         
