from util.starutil_numpy import radectoxyz, deg2distsq, distsq2arcsec
import spherematch as libkb_spherematch
#import astrometry

import numpy as np

def match(ra1, dec1, ra2, dec2,radius_in_deg):
    '''
    Wrapper arround spherematch.py (from libkd of astrometry.net)
    Behaves like spherematch.pro of IDL 
    (m1,m2,d12) = match(ra1,dec2, ra2,dec2, radius_in_deg)
    '''

    # Conver to coordinates on unit sphere
    xyz1 = radectoxyz(ra1, dec1)
    xyz2 = radectoxyz(ra2, dec2)
    r = np.sqrt(deg2distsq(radius_in_deg))

    # Run spherematch.py
    (inds,dists) =   libkb_spherematch.match(xyz1, xyz2, r)
    
    dist_in_deg = distsq2arcsec(np.power(dists,2))/3600.
    
    return  inds[:,0], inds[:,1], dist_in_deg
