#! /usr/bin/env python
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

from astrometry.util.siap import *
from astrometry.util.fits import *

if __name__ == '__main__':
    from optparse import OptionParser
    import sys
    parser = OptionParser('%prog <input vo-table> <output FITS.')
    opt,args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)
    T = siap_parse_result(args[0])
    T.writeto(args[1])
    
