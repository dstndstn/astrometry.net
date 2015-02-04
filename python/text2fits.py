#! /usr/bin/env python
import sys
from optparse import OptionParser
import numpy as np

from astrometry.util.fits import *

if __name__ == '__main__':
    p = OptionParser(usage='Usage: %prog [options] <input-text-file> <output-fits-table>')
    p.add_option('-s', dest='separator', help='Separator character (default: whitespace)')
    p.add_option('-S', dest='skiplines', type='int', help='Skip this number of lines before the header')
    p.add_option('-m', dest='maxcols', type='int', help='Trim each data row to this number of characters.')
    p.add_option('-H', dest='header', help='Header string containing column names')
    #p.add_option('-F', dest='floats', action='store_true', default=False,
    #            help='Assume all floats')
    p.add_option('-f', dest='format',
                 help='Formats: (f=float32, d=float64, s=string, j=int32, k=int64)')
    p.add_option('-n', dest='fnull', action='append', default=['null'],
                 help='Floating-point null value string (eg, "NaN", "null")')
    p.add_option('-N', dest='inull', action='append', default=['null'],
                 help='Integer null value string (eg, "null")')
    p.set_defaults(separator=None, maxcols=None, skiplines=0)
    (opt,args) = p.parse_args()
    if len(args) != 2:
        p.print_help()
        sys.exit(-1)
    textfn = args[0]
    fitsfn = args[1]

    coltypes = None
    #if opt.floats:
    #   coltypes = 
    if opt.format:
        typemap = {'d':np.float64, 'f':np.float32, 's':str,
                   'j':np.int32, 'k':np.int64}
        coltypes = [typemap[c] for c in opt.format]

    fnulls = dict([(x, np.nan) for x in opt.fnull])
    inulls = dict([(x, -1) for x in opt.inull])
    
    #T = text_table_fields(textfn, split=opt.separator, maxcols=opt.maxcols,
    #                      skiplines=opt.skiplines, headerline=opt.header,
    #                      coltypes=coltypes, floatvalmap=fnulls)
    fnulls[''] = np.nan
    T = streaming_text_table(textfn, split=opt.separator, maxcols=opt.maxcols,
                             skiplines=opt.skiplines, headerline=opt.header,
                             coltypes=coltypes, floatvalmap=fnulls,
                             intvalmap=inulls)
    T.write_to(fitsfn)

    
