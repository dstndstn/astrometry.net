import math
import os
import urllib

from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net import settings

def plot_sdss_image(wcsfn, plotfn):
    from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    # Parse the wcs.fits file
    wcs = anutil.Tan(wcsfn, 0)
    # arcsec radius
    #scale = math.hypot(wcs.imagew, wcs.imageh)/2. * wcs.pixel_scale()
    # grab SDSS tiles with about the same resolution as this image.
    logmsg('Image scale is', wcs.pixel_scale(), 'arcsec/pix')
    # size of SDSS image tiles to request, in pixels
    sdsssize = 512
    scale = sdsssize * wcs.pixel_scale() / 60.
    # healpix-vs-north-up rotation
    nside = anutil.healpix_nside_for_side_length_arcmin(scale / math.sqrt(2.))
    nside = 2 ** int(math.ceil(math.log(nside)/math.log(2.)))
    logmsg('Next power-of-2 nside:', nside)
    ra,dec = wcs.radec_center()
    logmsg('Image center is RA,Dec', ra, dec)

    dirnm = os.path.join(settings.SDSS_TILE_DIR, 'nside%i'%nside)
    if not os.path.exists(dirnm):
        os.makedirs(dirnm)

    #hp = anutil.radecdegtohealpix(ra, dec, nside)
    #logmsg('Healpix of center:', hp)
    radius = wcs.radius()
    hps = anutil.healpix_rangesearch_radec(ra, dec, radius, nside)
    logmsg('Healpixes in range:', hps)

    scale = math.sqrt(2.) * anutil.healpix_side_length_arcmin(nside) * 60. / float(sdsssize)
    logmsg('Grabbing SDSS tile with scale', scale, 'arcsec/pix')

    plot = ps.Plotstuff(outformat='png', wcsfn=wcsfn)
    img = plot.image
    img.format = ps.PLOTSTUFF_FORMAT_JPG
    img.resample = 1

    for hp in hps:
        fn = os.path.join(dirnm, '%i.jpg'%hp)
        logmsg('Checking for filename', fn)
        if not os.path.exists(fn):
            ra,dec = anutil.healpix_to_radecdeg(hp, nside, 0.5, 0.5)
            logmsg('Healpix center is RA,Dec', ra, dec)
            url = ('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?' +
                   'ra=%f&dec=%f&scale=%f&opt=&width=%i&height=%i' %
                   (ra, dec, scale, sdsssize, sdsssize))
            urllib.urlretrieve(url, fn)
            logmsg('Wrote', fn)
        swcsfn = os.path.join(dirnm, '%i.wcs'%hp)
        logmsg('Checking for WCS', swcsfn)
        if not os.path.exists(swcsfn):
            # Create WCS header
            cd = scale / 3600.
            swcs = anutil.Tan(ra, dec, sdsssize/2 + 0.5, sdsssize/2 + 0.5,
                              -cd, 0, 0, -cd, sdsssize, sdsssize)
            swcs.write_to(swcsfn)
            logmsg('Wrote WCS to', swcsfn)

        img.set_wcs_file(swcsfn, 0)
        img.set_file(fn)
        plot.plot('image')

    if False:
        out = plot.outline
        plot.color = 'white'
        plot.alpha = 0.25
        for hp in hps:
            swcsfn = os.path.join(dirnm, '%i.wcs'%hp)
            ps.plot_outline_set_wcs_file(out, swcsfn, 0)
            plot.plot('outline')

    plot.write(plotfn)
