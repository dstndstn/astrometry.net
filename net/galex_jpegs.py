import os

from astrometry.util.fits import fits_table
from astrometry.util.util import Tan
from astrometry.solver import plotstuff as ps
from astrometry.util.starutil_numpy import *

from astrometry.net.log import *

import numpy as np

def plot_into_wcs(wcsfn, plotfn, wcsext=0, basedir='.', scale=1.0):
    wcs = Tan(wcsfn, wcsext)
    ra,dec = wcs.radec_center()
    radius = wcs.radius()

    # The "index.fits" table has RA,Dec center and file paths.
    T = fits_table(os.path.join(basedir, 'index.fits'))
    # MAGIC 1: the GALEX fields are all smaller than 1 degree (~0.95) in radius,
    # so add that to the search 
    r = radius + 1.
    # find rows in "T" that are within range.
    J = points_within_radius(ra, dec, r, T.ra, T.dec)
    T = T[J]
    debug(len(T), 'GALEX fields within range of RA,Dec = ', ra, dec,
          'radius', radius)

    size = [int(scale*wcs.imagew),int(scale*wcs.imageh)]

    plot = ps.Plotstuff(outformat='png', wcsfn=wcsfn, wcsext=wcsext, size=size)
    plot.scale_wcs(scale)

    #debug('WCS:', str(plot.wcs))
    #plot.wcs.write_to('/tmp/wcs.fits')

    img = plot.image
    img.format = ps.PLOTSTUFF_FORMAT_JPG
    img.resample = 1
    if len(T):
        paths = [fn.strip() for fn in T.path]
    else:
        paths = []
        plot.color = 'black'
        plot.plot('fill')
    for jpegfn in paths:
        jpegfn = os.path.join(basedir, jpegfn)
        imwcsfn = jpegfn.replace('.jpg', '.wcs')
        debug('  Plotting GALEX fields', jpegfn, imwcsfn)
        #debug('jpeg "%s"' % jpegfn)
        #debug('wcs  "%s"' % imwcsfn)
        img.set_wcs_file(imwcsfn, 0)
        img.set_file(jpegfn)
        # convert black to transparent
        ps.plot_image_read(plot.pargs, img)
        ps.plot_image_make_color_transparent(img, 0, 0, 0)
        plot.plot('image')
    plot.write(plotfn)
    debug('wrote', plotfn)
