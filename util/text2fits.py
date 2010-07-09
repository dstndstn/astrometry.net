#! /usr/bin/env python
import sys
from optparse import OptionParser

from astrometry.util.pyfits_utils import *

if __name__ == '__main__':
	p = OptionParser(usage='Usage: %prog [options] <input-text-file> <output-fits-table>')
	p.add_option('-s', dest='separator', help='Separator character (default: whitespace)')
	p.add_option('-m', dest='maxcols', type='int', help='Trim each data row to this number of characters.')
	p.set_defaults(separator=None, maxcols=None)
	(opt,args) = p.parse_args()
	if len(args) != 2:
		p.print_help()
		sys.exit(-1)
	textfn = args[0]
	fitsfn = args[1]
	
	t = text_table_fields(textfn, split=opt.separator, maxcols=opt.maxcols)
	ft = pyfits.new_table(t.to_fits_columns())
	ft.writeto(fitsfn, clobber=True)

	
