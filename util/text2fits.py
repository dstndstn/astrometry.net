#! /usr/bin/env python
import sys

from astrometry.util.pyfits_utils import *

if __name__ == '__main__':
	args = sys.argv[1:]
	if len(args) != 2:
		print 'Usage: %s <input-text-file> <output-fits-table>'
		sys.exit(-1)
	textfn = args[0]
	fitsfn = args[1]
	
	t = text_table_fields(textfn)
	ft = pyfits.new_table(t.to_fits_columns())
	ft.writeto(fitsfn, clobber=True)

	
