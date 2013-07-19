import numpy as np
import scipy.interpolate as interp
from miscutils import lanczos_filter

# DEBUG
from astrometry.util.plotutils import *
import pylab as plt

def resample_with_wcs(targetwcs, wcs, Limages, L, spline=True):
    '''
    Returns (Yo,Xo, Yi,Xi, ims)

    or (None,None,None,None,None) if the target and input wcs do not
    overlap enough.

    Limages: list of images to Lanczos-interpolate at the given Lanczos order.
    If empty, just returns nearest-neighbour indices.

    L: int, lanczos order

    targetwcs, wcs: duck-typed WCS objects that must have:
       - properties "imagew", "imageh"
       - methods  "r,d = pixelxy2radec(x, y)"
       -          "ok,x,y = radec2pixelxy(ra, dec)"

    The WCS functions are expected to operate in FITS pixel-indexing.

    Use the results like:

    target[Yo,Xo] = nearest_neighbour[Yi,Xi]

    target[Yo,Xo] = ims[i]
    
    '''
    # Adapted from detection/sdss-demo.py

    ### DEBUG
    #ps = PlotSequence('resample')
    ps = None

    H,W = int(targetwcs.imageh), int(targetwcs.imagew)
    h,w = int(      wcs.imageh), int(      wcs.imagew)

    for im in Limages:
        assert(im.shape == (h,w))

    #print 'Target size', W, H
    #print 'Input size', w, h
    
    # First find the approximate bbox of the input image in
    # the target image so that we don't ask for way too
    # many out-of-bounds pixels...
    XY = []
    for x,y in [(0,0), (w-1,0), (w-1,h-1), (0, h-1)]:
        ra,dec = wcs.pixelxy2radec(float(x + 1), float(y + 1))
        ok,xw,yw = targetwcs.radec2pixelxy(ra, dec)
        XY.append((xw - 1, yw - 1))
    XY = np.array(XY)

    x0,y0 = np.round(XY.min(axis=0)).astype(int)
    x1,y1 = np.round(XY.max(axis=0)).astype(int)
    if spline:
        # Now we build a spline that maps "target" pixels to "input" pixels
        # spline inputs: pixel coords in the 'target' image
        margin = 20
        step = 25
        xlo = max(0, x0-margin)
        xhi = min(W, x1+margin)
        nx = np.ceil(float(xhi - xlo) / step) + 1
        xx = np.linspace(xlo, xhi, nx)
        ylo = max(0, y0-margin)
        yhi = min(H, y1+margin)
        ny = np.ceil(float(yhi - ylo) / step) + 1
        yy = np.linspace(ylo, yhi, ny)

        if ps:
            def expand_axes():
                M = 100
                ax = plt.axis()
                plt.axis([ax[0]-M, ax[1]+M, ax[2]-M, ax[3]+M])
                plt.axis('scaled')

            plt.clf()
            plt.plot(XY[:,0], XY[:,1], 'ro')
            plt.plot(xx, np.zeros_like(xx), 'b.')
            plt.plot(np.zeros_like(yy), yy, 'c.')
            plt.plot(xx, np.zeros_like(xx)+max(yy), 'b.')
            plt.plot(max(xx) + np.zeros_like(yy), yy, 'c.')
            plt.plot([0,W,W,0,0], [0,0,H,H,0], 'k-')
            plt.title('A: Target image: bbox')
            expand_axes()
            ps.savefig()

        if (len(xx) == 0) or (len(yy) == 0):
            print 'No overlap between input and target WCSes'
            return (None,)*5
        if (len(xx) <= 3) or (len(yy) <= 3):
            print 'Not enough overlap between input and target WCSes'
            return (None,)*5
    
        # spline outputs -- pixel coords in the 'input' image
        XYo = []
        for y in yy:
            for x in xx:
                #rd = targetwcs.pixelToPosition(x,y)
                #XYo.append(wcs.positionToPixel(rd))
                ra,dec = targetwcs.pixelxy2radec(float(x + 1), float(y + 1))
                ok,xw,yw = wcs.radec2pixelxy(ra,dec)
                XYo.append((xw - 1, yw - 1))
        XYo = np.array(XYo)
        Xo = XYo[:,0].reshape(len(yy), len(xx))
        Yo = XYo[:,1].reshape(len(yy), len(xx))
        del XYo

        if ps:
            plt.clf()
            plt.plot(Xo, Yo, 'b.')
            plt.plot([0,w,w,0,0], [0,0,h,h,0], 'k-')
            plt.title('B: Input image')
            expand_axes()
            ps.savefig()
    
        xspline = interp.RectBivariateSpline(xx, yy, Xo.T)
        yspline = interp.RectBivariateSpline(xx, yy, Yo.T)
        del Xo
        del Yo

    else:
        margin = 0

    # Now, build the full pixel grid we want to interpolate...
    ixo = np.arange(max(0, x0-margin), min(W, x1+margin+1), dtype=int)
    iyo = np.arange(max(0, y0-margin), min(H, y1+margin+1), dtype=int)

    if len(ixo) == 0 or len(iyo) == 0:
        return (None,)*5

    if spline:
        # And run the interpolator.  [xy]spline() does a meshgrid-like broadcast,
        # so fxi,fyi have shape n(iyo),n(ixo)
        # f[xy]i: floating-point pixel coords in the input image
        fxi = xspline(ixo, iyo).T
        fyi = yspline(ixo, iyo).T

        if ps:
            plt.clf()
            plt.plot(ixo, np.zeros_like(ixo), 'r,')
            plt.plot(np.zeros_like(iyo), iyo, 'm,')
            plt.plot(ixo, max(iyo) + np.zeros_like(ixo), 'r,')
            plt.plot(max(ixo) + np.zeros_like(iyo), iyo, 'm,')
            plt.plot([0,W,W,0,0], [0,0,H,H,0], 'k-')
            plt.title('C: Target image; i*o')
            expand_axes()
            ps.savefig()
    
            plt.clf()
            plt.plot(fxi, fyi, 'r,')
            plt.plot([0,w,w,0,0], [0,0,h,h,0], 'k-')
            plt.title('D: Input image, f*i')
            expand_axes()
            ps.savefig()

    else:
        fxi = np.empty((len(iyo),len(ixo)))
        fyi = np.empty((len(iyo),len(ixo)))

        fxo = (ixo).astype(float) + 1.
        fyo = np.empty_like(fxo)
        for i,y in enumerate(iyo):
            fyo[:] = y + 1.
            # Assume 1-d vectorized pixel<->radec
            ra,dec = targetwcs.pixelxy2radec(fxo, fyo)
            ok,x,y = wcs.radec2pixelxy(ra, dec)
            fxi[i,:] = x - 1.
            fyi[i,:] = y - 1.

    # print 'ixo', ixo.shape
    # print 'iyo', iyo.shape
    # print 'fxi', fxi.shape
    # print 'fyi', fyi.shape

    # i[xy]i: int coords in the input image
    ixi = np.round(fxi).astype(int)
    iyi = np.round(fyi).astype(int)

    # Keep only in-bounds pixels.
    I = np.flatnonzero((ixi >= 0) * (iyi >= 0) * (ixi < w) * (iyi < h))
    fxi = fxi.flat[I]
    fyi = fyi.flat[I]
    ixi = ixi.flat[I]
    iyi = iyi.flat[I]
    #print 'I', I.shape
    #print 'dims', (len(iyo),len(ixo))
    iy,ix = np.unravel_index(I, (len(iyo),len(ixo)))
    iyo = iyo[0] + iy
    ixo = ixo[0] + ix
    #ixo = ixo[I % len(ixo)]
    #iyo = iyo[I / len(ixo)]
    # i[xy]o: int coords in the target image


    if spline and ps:
        plt.clf()
        plt.plot(ixo, iyo, 'r,')
        plt.plot([0,W,W,0,0], [0,0,H,H,0], 'k-')
        plt.title('E: Target image; i*o')
        expand_axes()
        ps.savefig()

        plt.clf()
        plt.plot(fxi, fyi, 'r,')
        plt.plot([0,w,w,0,0], [0,0,h,h,0], 'k-')
        plt.title('F: Input image, f*i')
        expand_axes()
        ps.savefig()

    assert(np.all(ixo >= 0))
    assert(np.all(iyo >= 0))
    assert(np.all(ixo < W))
    assert(np.all(iyo < H))

    assert(np.all(ixi >= 0))
    assert(np.all(iyi >= 0))
    assert(np.all(ixi < w))
    assert(np.all(iyi < h))

    if len(Limages):
        fxi -= ixi
        fyi -= iyi
        dx = fxi
        dy = fyi
        del fxi
        del fyi

        # Lanczos interpolation.
        # number of pixels
        nn = len(ixo)
        NL = 2*L+1

        # We interpolate all the pixels at once.

        # accumulators for each input image
        laccs = [np.zeros(nn) for im in Limages]
        # sum of lanczos terms
        fsum = np.zeros(nn)

        off = np.arange(-L, L+1)
        for oy in off:
            fy = lanczos_filter(L, -oy + dy)
            for ox in off:
                fx = lanczos_filter(L, -ox + dx)
                for lacc,im in zip(laccs, Limages):
                    lacc += fx * fy * im[np.clip(iyi + oy, 0, h-1),
                                         np.clip(ixi + ox, 0, w-1)]
                fsum += fx*fy
        for lacc in laccs:
            lacc /= fsum

        rims = laccs

    else:
        rims = []

    return (iyo,ixo, iyi,ixi, rims)
