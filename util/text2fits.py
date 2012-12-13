#! /usr/bin/env python
import sys
from optparse import OptionParser
import numpy as np

from astrometry.util.pyfits_utils import *

if __name__ == '__main__':
	p = OptionParser(usage='Usage: %prog [options] <input-text-file> <output-fits-table>')
	p.add_option('-s', dest='separator', help='Separator character (default: whitespace)')
	p.add_option('-S', dest='skiplines', type='int', help='Skip this number of lines before the header')
	p.add_option('-m', dest='maxcols', type='int', help='Trim each data row to this number of characters.')
	p.add_option('-H', dest='header', help='Header string containing column names')
	#p.add_option('-F', dest='floats', action='store_true', default=False,
	#			 help='Assume all floats')
	p.add_option('-f', dest='format',
				 help='Formats: (f=float32, d=float64)')
	p.set_defaults(separator=None, maxcols=None, skiplines=0)
	(opt,args) = p.parse_args()
	if len(args) != 2:
		p.print_help()
		sys.exit(-1)
	textfn = args[0]
	fitsfn = args[1]

	coltypes = None
	#if opt.floats:
	#	coltypes = 
	if opt.format:
		coltypes = [{'d':np.float64, 'f':np.float32}[c] for c in opt.format]
	
	t = text_table_fields(textfn, split=opt.separator, maxcols=opt.maxcols,
						  skiplines=opt.skiplines, headerline=opt.header,
						  coltypes=coltypes)
	t.write_to(fitsfn)

	
