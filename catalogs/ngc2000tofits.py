# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function

from astrometry.util.fits import fits_table

from ngc2000 import ngc2000, ngc2000accurate
from astrometry.util.fits import *

if __name__ == '__main__':

    T = fits_table()
    for key in ['is_ngc', 'ra', 'dec', 'size', 'classification']:
        T.set(key, [x[key] for x in ngc2000])
    # special keyword
    T.num = [x['id'] for x in ngc2000]
    T.to_np_arrays()

    # update with ngc2000-accurate positions
    # build map to destination indices
    imap = dict([((isit,n), i) for i,(isit,n) in
                 enumerate(zip(T.is_ngc, T.num))])

    nup = 0
    for x in ngc2000accurate:
        key = (x['is_ngc'], x['id'])
        try:
            i = imap[key]
        except KeyError:
            continue
        T.ra [i] = x['ra']
        T.dec[i] = x['dec']
        nup +=1
    print('updated %i' % nup)

    # turn from diameter in arcmin to radius in deg.
    T.radius = T.size / (2. * 60.)
    T.delete_column('size')

    T.name = np.array(['NGC %i' % n if isngc else 'IC %i' % n
                       for n,isngc in zip(T.num, T.is_ngc)])

    units_dict = dict(ra='deg', dec='deg', radius='deg')
    units = [units_dict.get(c, None) for c in T.get_columns()]

    for k in ['ra','dec','radius']:
        T.set(k, T.get(k).astype(np.float32))
    T.num = T.num.astype(np.int16)

    NGC = T[T.is_ngc]
    NGC.rename('num', 'ngcnum')
    NGC.delete_column('is_ngc')
    units = [units_dict.get(c, '') for c in NGC.get_columns()]
    NGC.writeto('ngc2000.fits', units=units)

    IC = T[np.logical_not(T.is_ngc)]
    IC.rename('num', 'icnum')
    IC.delete_column('is_ngc')
    units = [units_dict.get(c, '') for c in IC.get_columns()]
    IC.writeto('ic2000.fits', units=units)
    
