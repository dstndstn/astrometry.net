# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from astrometry.util.fits import *
from numpy import logical_and

import optparse
import sys

if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] <input filename (2mass_hpXXX.fits)> <output-filename>')
    parser.add_option('-b', dest='band', help='Select the band on which to apply cuts: "J" (default), "H", or "K"')
    parser.set_defaults(band='J')

    opt,args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)

    lband = opt.band.lower()
    infn = args[0]
    outfn = args[1]

    print('Reading %s, writing %s' % (infn, outfn))
    T = fits_table(infn)
    qual_col = '%s_quality' % lband
    cc_col = '%s_cc' % lband
    mag_col = '%s_mag' % lband
    qual = T.getcolumn(qual_col)
    cc = T.getcolumn(cc_col)

    # if ((entry->j_quality != TWOMASS_QUALITY_NO_BRIGHTNESS) &&
    #     (entry->j_cc == TWOMASS_CC_NONE)) {
        
    I = logical_and(qual != chr(0), cc == chr(0))
    print('Keeping %i of %i sources' % (sum(I), len(I)))

    T[I].write_to(outfn, columns=['ra','dec',mag_col])


