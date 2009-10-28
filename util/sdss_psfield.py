from astrometry.util.pyfits_utils import *
from pyfits import *

# Returns "gain", "dark_variance", and "skyErr"
def sdss_psfield_noise(psfieldfn, band=None):
	p = table_fields(pyfits.open(psfieldfn)[6].data)
	p = p[0]
	if band is not None:
		return (p.gain[band], p.dark_variance[band], p.sky[band], p.skyerr[band])
	return (p.gain, p.dark_variance, p.sky, p.skyerr)
