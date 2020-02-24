from __future__ import print_function
import math
import os
try:
    # py3
    from urllib.request import urlretrieve
except ImportError:
    # py2
    from urllib import urlretrieve

if __name__ == '__main__':
    import os
    os.environ['DJANGO_SETTINGS_MODULE'] = 'astrometry.net.settings'

from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net import settings

def plot_sdss_image(wcsfn, plotfn, image_scale=1.0, debug_ps=None):
    from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    # Parse the wcs.fits file
    wcs = anutil.Tan(wcsfn, 0)
    # grab SDSS tiles with about the same resolution as this image.
    pixscale = wcs.pixel_scale()
    pixscale = pixscale / image_scale
    logmsg('Original image scale is', wcs.pixel_scale(), 'arcsec/pix; scaled', image_scale, '->', pixscale)
    # size of SDSS image tiles to request, in pixels
    sdsssize = 512
    scale = sdsssize * pixscale / 60.
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
    logmsg('Healpixes in range:', len(hps), ': ', hps)

    scale = math.sqrt(2.) * anutil.healpix_side_length_arcmin(nside) * 60. / float(sdsssize)
    logmsg('Grabbing SDSS tile with scale', scale, 'arcsec/pix')

    size = [int(image_scale*wcs.imagew),int(image_scale*wcs.imageh)]

    plot = ps.Plotstuff(outformat='png', wcsfn=wcsfn, size=size)
    plot.scale_wcs(image_scale)
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
            urlretrieve(url, fn)
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

        if debug_ps is not None:
            fn = debug_ps.getnext()
            plot.write(fn)
            print('Wrote', fn)
        
    if debug_ps is not None:
        out = plot.outline
        plot.color = 'white'
        plot.alpha = 0.25
        for hp in hps:
            swcsfn = os.path.join(dirnm, '%i.wcs'%hp)
            ps.plot_outline_set_wcs_file(out, swcsfn, 0)
            plot.plot('outline')
        plot.write(fn)
        print('Wrote', fn)

    plot.write(plotfn)




if __name__ == '__main__':
    import logging
    from astrometry.util import util as anutil

    logging.basicConfig(format='%(message)s',
                        level=logging.DEBUG)

    wcsfn = 'wcs.fits'
    outfn = 'sdss.png'

    if True:
        wcs = anutil.Tan(wcsfn)
        scale = 640. / wcs.get_width()
        print('Scale', scale)
        
    from astrometry.util.plotutils import *

    ps = PlotSequence('sdss')
    
    plot_sdss_image(wcsfn, outfn, image_scale=scale, debug_ps=ps)
    
