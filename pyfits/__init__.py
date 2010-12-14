#This is the configuration file for the pyfits namespace.

from __future__ import division # confidence high

import os

# Define the version of the pyfits package.
try:
    import svn_version
    __svn_version__ = svn_version.__svn_version__
except ImportError:
    __svn_version__ = 'Unable to determine SVN revision'

__version__ = '2.3.1'

# Import the pyfits core module.
from core import *
__doc__ = core.__doc__

# Define modules available using from pyfits import *.
_locals = locals().keys()
for n in _locals[::-1]:
    if n[0] == '_' or n in ('re', 'os', 'tempfile', 'exceptions', 'operator', 'num', 'ndarray', 'chararray', 'rec', 'objects', 'Memmap', 'maketrans', 'open'):
        _locals.remove(n)
__all__ = _locals

try:
    import pytools.tester
    def test(*args,**kwds):
        pytools.tester.test(modname=__name__, *args, **kwds)
except ImportError:
    pass

