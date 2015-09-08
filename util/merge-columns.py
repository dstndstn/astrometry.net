#! /usr/bin/env python
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE

from astrometry.util.fits import *
import sys

if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser(prog='[input files] [output file]')
    opt,args = parser.parse_args()

    if len(args) < 2:
        parser.print_usage()
        sys.exit(-1)

    infns = args[:-1]
    outfn = args[-1]

    print 'Writing to output file', outfn
    T = None
    for fn in infns:
        print 'Reading input', fn
        Ti = fits_table(fn)
        if T is None:
            T = Ti
        else:
            T.add_columns_from(Ti)
    T.writeto(outfn)
    

    
    
