#This is the configuration file for the pyfits namespace.  This is needed
#because we have the option of using either a numarray or numpy version
#of pyfits.

#This option is controlled by the NUMERIX environment variable.  Set NUMERIX 
#to 'numarray' for the numarray version of pyfits.  Set NUMERIX to 'numpy'
#for the numpy version of pyfits.

#If only one array package is installed, that package's version of pyfits
#will be imported.  If both packages are installed the NUMERIX value is
#used to decide between the packages.  If no NUMERIX value is set then 
#the numpy version of pyfits will be imported.

#Anything else is an exception.

import os

__version__ = '2.0.1dev424'

# Check the environment variables for NUMERIX
try:
    numerix = os.environ["NUMERIX"]
except:
    numerix = 'numpy'


if (numerix == 'numarray'):
    try :
        from NA_pyfits import *
        import NA_pyfits as core
        __doc__ = NA_pyfits.__doc__
    except ImportError, e:
        raise ImportError, `e` + ".  Cannot import numarray version of PyFITS!"
else:
    try:
        try:
            from NP_pyfits import *
            import NP_pyfits as core
            __doc__ = NP_pyfits.__doc__
        except ImportError:
            try:
                from NA_pyfits import *
                import NA_pyfits as core
                doc__ = NA_pyfits.__doc__
            except ImportError, e:
                raise ImportError, `e` + ".  Cannot import either numpy or numarray."
    except Exception, e:
        raise ImportError, `e` + ".  No usable array package has been found.  Cannot import either numpy or numarray."
    
_locals = locals().keys()
for n in _locals[::-1]:
    if n[0] == '_' or n in ('re', 'os', 'tempfile', 'exceptions', 'operator', 'num', 'ndarray', 'chararray', 'rec', 'objects', 'Memmap', 'maketrans', 'open'):
        _locals.remove(n)
__all__ = _locals
