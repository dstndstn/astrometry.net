# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from astrometry.util.resample import resample_with_wcs, ResampleError

from .fields import radec_to_sdss_rcf
from .common import band_name, band_index, AsTransWrapper
from functools import reduce

def get_sdss_cutout(targetwcs, sdss, get_rawvals=False, bands='irg',
                    get_rawvals_only=False,
                    bandscales=dict(z=1.0, i=1.0, r=1.3, g=2.5)):

    rgbims = []

    ra,dec = targetwcs.radec_center()
    # in deg
    radius = targetwcs.radius()
    #print 'Target WCS radius is', radius, 'deg'
    H,W = targetwcs.get_height(), targetwcs.get_width()
    targetpixscale = targetwcs.pixel_scale()
    
    wlistfn = sdss.filenames.get('window_flist', 'window_flist.fits')
    rad2 = radius*60. + np.hypot(14., 10.)/2.
    #print 'Rad2 radius', rad2, 'arcmin'
    RCF = radec_to_sdss_rcf(ra, dec, tablefn=wlistfn, radius=rad2)

    # Drop rerun 157
    keepRCF = []
    for run,camcol,field,r,d in RCF:
        rr = sdss.get_rerun(run, field)
        #print 'Rerun:', rr
        if rr == '157':
            continue
        keepRCF.append((run,camcol,field))
    RCF = keepRCF
    print(len(RCF), 'run/camcol/fields in range')

    # size in SDSS pixels of the target image.
    sz = np.hypot(H, W)/2. * targetpixscale / 0.396
    print('SDSS sz:', sz)

    bandnums = [band_index(b) for b in bands]
    
    for bandnum,band in zip(bandnums, bands):
        targetim = np.zeros((H, W), np.float32)
        targetn  = np.zeros((H, W), np.uint8)

        for ifield,(run,camcol,field) in enumerate(RCF):
            
            fn = sdss.retrieve('frame', run, camcol, field, band)
            frame = sdss.readFrame(run, camcol, field, bandnum)

            h,w = frame.getImageShape()
            x,y = frame.astrans.radec_to_pixel(ra, dec)
            x,y = int(x), int(y)
            # add some margin for resampling
            sz2 = int(sz) + 5
            xlo = np.clip(x - sz2, 0, w)
            xhi = np.clip(x + sz2 + 1, 0, w)
            ylo = np.clip(y - sz2, 0, h)
            yhi = np.clip(y + sz2 + 1, 0, h)
            if xlo == xhi or ylo == yhi:
                continue
            stamp = frame.getImageSlice((slice(ylo, yhi), slice(xlo, xhi)))
            sh,sw = stamp.shape
            wcs = AsTransWrapper(frame.astrans, sw, sh, x0=xlo, y0=ylo)
            # FIXME -- allow nn resampling too
            try:
                Yo,Xo,Yi,Xi,[rim] = resample_with_wcs(targetwcs, wcs, [stamp], 3)
            except ResampleError:
                continue
            targetim[Yo,Xo] += rim
            targetn [Yo,Xo] += 1

        rgbims.append(targetim / targetn)

    if get_rawvals_only:
        return rgbims

    if get_rawvals:
        rawvals = [x.copy() for x in rgbims]

    r,g,b = rgbims

    r *= bandscales[bands[0]]
    g *= bandscales[bands[1]]
    b *= bandscales[bands[2]]
    
    # i
    #r *= 1.0
    # r
    #g *= 1.5
    #g *= 1.3
    # g
    #b *= 2.5
    m = -0.02
    r = np.maximum(0, r - m)
    g = np.maximum(0, g - m)
    b = np.maximum(0, b - m)
    I = (r+g+b)/3.
    alpha = 1.5
    Q = 20
    m2 = 0.
    fI = np.arcsinh(alpha * Q * (I - m2)) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    R = fI * r / I
    G = fI * g / I
    B = fI * b / I
    maxrgb = reduce(np.maximum, [R,G,B])
    J = (maxrgb > 1.)
    R[J] = R[J]/maxrgb[J]
    G[J] = G[J]/maxrgb[J]
    B[J] = B[J]/maxrgb[J]
    ss = 0.5
    RGBblur = np.clip(np.dstack([
        gaussian_filter(R, ss),
        gaussian_filter(G, ss),
        gaussian_filter(B, ss)]), 0., 1.)

    if get_rawvals:
        return RGBblur, rawvals
    return RGBblur
        


if __name__ == '__main__':
    import tempfile
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    from .dr10 import DR10
    from astrometry.util.util import Tan
    
    tempdir = tempfile.gettempdir()

    sdss = DR10(basedir=tempdir)
    sdss.saveUnzippedFiles(tempdir)

    W,H = 100, 100
    pixscale = 1.
    cd = pixscale / 3600.
    targetwcs = Tan(120., 10., W/2., H/2., -cd, 0., 0., cd, float(W), float(H))
    rgb = get_sdss_cutout(targetwcs, sdss)

    plt.clf()
    plt.imshow(rgb, interpolation='nearest', origin='lower')
    plt.savefig('cutout1.png')
    

    W,H = 3000, 3000
    pixscale = 0.5
    cd = pixscale / 3600.
    targetwcs = Tan(120., 10., W/2., H/2., -cd, 0., 0., cd, float(W), float(H))
    rgb = get_sdss_cutout(targetwcs, sdss)

    plt.clf()
    plt.imshow(rgb, interpolation='nearest', origin='lower')
    plt.savefig('cutout2.png')
    
