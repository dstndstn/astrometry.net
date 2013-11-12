import numpy as np
import scipy.interpolate as interp
from miscutils import lanczos_filter

# DEBUG
from astrometry.util.plotutils import *
import pylab as plt

class ResampleError(Exception):
    pass
class OverlapError(ResampleError):
    pass
class NoOverlapError(OverlapError):
    pass
class SmallOverlapError(OverlapError):
    pass

def resample_with_wcs(targetwcs, wcs, Limages, L, spline=True,
                      splineFallback = True,
                      splineStep = 25,
                      splineMargin = 12,
                      table=True,
                      cinterp = True):
    '''
    Returns (Yo,Xo, Yi,Xi, ims)

    Use the results like:

    target[Yo,Xo] = nearest_neighbour[Yi,Xi]
    # or
    target[Yo,Xo] = ims[i]


    raises NoOverlapError if the target and input WCSes do not
    overlap.  Raises SmallOverlapError if they do not overlap "enough"
    (as described below).

    targetwcs, wcs: duck-typed WCS objects that must have:
       - properties "imagew", "imageh"
       - methods  "r,d = pixelxy2radec(x, y)"
       -          "ok,x,y = radec2pixelxy(ra, dec)"

    The WCS functions are expected to operate in FITS pixel-indexing.

    The WCS function must support 1-d, broadcasting, vectorized
    pixel<->radec calls.

    Limages: list of images to Lanczos-interpolate at the given Lanczos order.
    If empty, just returns nearest-neighbour indices.

    L: int, lanczos order

    spline: bool: use a spline interpolator to reduce the number of
    WCS calls.

    splineFallback: bool: the spline requires a certain amount of
    spatial overlap.  With splineFallback = True, fall back to
    non-spline version.  With splineFallback = False, just raises
    SmallOverlapError.

    splineStep: approximate grid size

    table: use Lanczos3 look-up table?

    '''
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
		# [-2:]: handle ok,ra,dec or ra,dec
        ok,xw,yw = targetwcs.radec2pixelxy(
			*(wcs.pixelxy2radec(float(x + 1), float(y + 1))[-2:]))
        XY.append((xw - 1, yw - 1))
    XY = np.array(XY)

    x0,y0 = np.round(XY.min(axis=0)).astype(int)
    x1,y1 = np.round(XY.max(axis=0)).astype(int)
    if spline:
        # Now we build a spline that maps "target" pixels to "input" pixels
        # spline inputs: pixel coords in the 'target' image
        margin = splineMargin
        step = splineStep
        xlo = max(0, x0-margin)
        xhi = min(W, x1+margin+1)
        nx = np.ceil(float(xhi - xlo) / step) + 1
        xx = np.linspace(xlo, xhi, nx)
        ylo = max(0, y0-margin)
        yhi = min(H, y1+margin+1)
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
            #print 'No overlap between input and target WCSes'
            raise NoOverlapError()

        if (len(xx) <= 3) or (len(yy) <= 3):
            #print 'Not enough overlap between input and target WCSes'
            if splineFallback:
                spline = False
            else:
                raise SmallOverlapError()

    if spline:
        # spline inputs  -- pixel coords in the 'target' image
        #    (xx, yy)
        # spline outputs -- pixel coords in the 'input' image
        #    (XX, YY)
        # We use vectorized radec <-> pixelxy functions here
        ok,XX,YY = wcs.radec2pixelxy(
            *(targetwcs.pixelxy2radec(
                xx[np.newaxis,:] + 1,
				yy[:,np.newaxis] + 1)[-2:]))
        XX -= 1.
        YY -= 1.
        del ok

        if ps:
            plt.clf()
            plt.plot(Xo, Yo, 'b.')
            plt.plot([0,w,w,0,0], [0,0,h,h,0], 'k-')
            plt.title('B: Input image')
            expand_axes()
            ps.savefig()
    
        xspline = interp.RectBivariateSpline(xx, yy, XX.T)
        yspline = interp.RectBivariateSpline(xx, yy, YY.T)
        del XX
        del YY

    else:
        margin = 0

    # Now, build the full pixel grid (in the ouput image) we want to
    # interpolate...
    ixo = np.arange(max(0, x0-margin), min(W, x1+margin+1), dtype=int)
    iyo = np.arange(max(0, y0-margin), min(H, y1+margin+1), dtype=int)

    if len(ixo) == 0 or len(iyo) == 0:
        raise NoOverlapError()

    if spline:
        # And run the interpolator.
        # [xy]spline() does a meshgrid-like broadcast, so fxi,fyi have
        # shape n(iyo),n(ixo)
        #
        # f[xy]i: floating-point pixel coords in the input image
        fxi = xspline(ixo, iyo).T.astype(np.float32)
        fyi = yspline(ixo, iyo).T.astype(np.float32)

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
        # Use 2-d broadcasting pixel <-> radec functions here.
        # This can be rather expensive, with lots of WCS calls!
        ok,fxi,fyi = wcs.radec2pixelxy(
            *targetwcs.pixelxy2radec(ixo[np.newaxis,:] + 1.,
                                     iyo[:,np.newaxis] + 1.))
        del ok
        fxi -= 1.
        fyi -= 1.
        
    # print 'ixo', ixo.shape
    # print 'iyo', iyo.shape
    # print 'fxi', fxi.shape
    # print 'fyi', fyi.shape

    # Keep only in-bounds pixels.
    ## HACK -- 0.51
    I = np.flatnonzero((fxi >= -0.5) * (fyi >= -0.5) *
                       (fxi < w-0.51) * (fyi < h-0.51))
    fxi = fxi.flat[I]
    fyi = fyi.flat[I]
    # i[xy]i: int coords in the input image
    ixi = np.round(fxi).astype(np.int32)
    iyi = np.round(fyi).astype(np.int32)

    #print 'dims', (len(iyo),len(ixo))
    iy,ix = np.unravel_index(I, (len(iyo),len(ixo)))
    iyo = iyo[0] + iy
    ixo = ixo[0] + ix
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
        dx = (fxi - ixi).astype(np.float32)
        dy = (fyi - iyi).astype(np.float32)
        del fxi
        del fyi
        # print 'dx', dx.min(), dx.max()
        # print 'dy', dy.min(), dy.max()

        # Lanczos interpolation.
        # number of pixels
        nn = len(ixo)
        NL = 2*L+1
        # accumulators for each input image
        laccs = [np.zeros(nn, np.float32) for im in Limages]

        if cinterp:
            from util import lanczos3_interpolate
            # ixi = ixi.astype(np.int)
            # iyi = iyi.astype(np.int)
            # print 'ixi/iyi', ixi.shape, ixi.dtype, iyi.shape, iyi.dtype
            # print 'dx/dy', dx.shape, dx.dtype, dy.shape, dy.dtype
            rtn = lanczos3_interpolate(ixi, iyi, dx, dy, laccs,
                                       [lim.astype(np.float32)
                                        for lim in Limages])
            # print 'rtn:', rtn
        else:
            _lanczos_interpolate(L, ixi, iyi, dx, dy, laccs, Limages,
                                 table=table)
        rims = laccs
    else:
        rims = []

    return (iyo,ixo, iyi,ixi, rims)


def _lanczos_interpolate(L, ixi, iyi, dx, dy, laccs, limages,
                         table=True):
    '''
    L: int, Lanczos order
    ixi: int, 1-d numpy array, len n, x coord in input images
    iyi:     ----""----        y
    dx: float, 1-d numpy array, len n, fractional x coord
    dy:      ----""----                    y
    laccs: list of [float, 1-d numpy array, len n]: outputs
    limages list of [float, 2-d numpy array, shape h,w]: inputs
    '''

    lfunc = lanczos_filter
    if L == 3:
        try:
            from util import lanczos3_filter, lanczos3_filter_table
            # 0: no rangecheck
            if table:
                #lfunc = lambda nil,x,y: lanczos3_filter_table(x,y, 0)
                lfunc = lambda nil,x,y: lanczos3_filter_table(x,y, 1)
            else:
                lfunc = lambda nil,x,y: lanczos3_filter(x,y)
        except:
            pass

    h,w = limages[0].shape
    n = len(ixi)
    # sum of lanczos terms
    fsum = np.zeros(n)
    off = np.arange(-L, L+1)
    #fx = np.zeros(n)
    #fy = np.zeros(n)
    fx = np.zeros(n, np.float32)
    fy = np.zeros(n, np.float32)
    for oy in off:
        #print 'dy range:', min(-oy + dy), max(-oy + dy)
        lfunc(L, -oy + dy, fy)
        for ox in off:
            lfunc(L, -ox + dx, fx)
            #print 'dx range:', min(-ox + dx), max(-ox + dx)
            for lacc,im in zip(laccs, limages):
                lacc += fx * fy * im[np.clip(iyi + oy, 0, h-1),
                                     np.clip(ixi + ox, 0, w-1)]
                fsum += fx*fy
    for lacc in laccs:
        lacc /= fsum



if __name__ == '__main__':
    import fitsio
    from astrometry.util.util import Sip,Tan
    import time
    import sys

    from astrometry.util.util import lanczos3_filter, lanczos3_filter_table
    # x = np.linspace(-4, 4, 500)
    # L = np.zeros_like(x)
    # L2 = np.zeros(len(x), np.float32)
    # lanczos3_filter(x, L)
    # lanczos3_filter_table(x.astype(np.float32), L2, 1)
    # plt.clf()
    # plt.plot(x, L, 'r-')
    # plt.plot(x, L2, 'b-')
    # plt.savefig('l1.png')

    x = np.linspace(-3.5, 4.5, 8192).astype(np.float32)
    L1 = np.zeros_like(x)
    L2 = np.zeros_like(x)
    lanczos3_filter(x, L1)
    lanczos3_filter_table(x, L2, 1)
    print 'L2 - L1 RMS:', np.sqrt(np.mean((L2-L1)**2))
    
    if True:
        ra,dec = 0.,0.,
        pixscale = 1e-3
        W,H = 10,1

        cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                    -pixscale, 0., 0., pixscale, W, H)
        dx,dy = 0.25, 0.
        wcs = Tan(ra, dec, (W+1)/2. + dx, (H+1)/2. + dy,
                  -pixscale, 0., 0., pixscale, W, H)

        pix = np.zeros((H,W), np.float32)
        pix[0,W/2] = 1.
        
        Yo,Xo,Yi,Xi,(cpix,) = resample_with_wcs(cowcs, wcs, [pix], 3)
        print 'C', cpix
        Yo2,Xo2,Yi2,Xi2,(pypix,) = resample_with_wcs(cowcs, wcs, [pix], 3, cinterp=False, table=False)
        print 'Py', pypix

        print 'RMS', np.sqrt(np.mean((cpix - pypix)**2))
        
        sys.exit(0)
        
        
    if True:
        ra,dec = 219.577111, 54.52
        pixscale = 2.75 / 3600.
        W,H = 10,10
        cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                    -pixscale, 0., 0., pixscale, W, H)

        for i,(dx,dy) in enumerate([(0.01, 0.02),
                                    (0.1, 0.0),
                                    (0.2, 0.0),
                                    (0.3, 0.0),
                                    (0.4, 0.0),
                                    (0.5, 0.0),
                                    (0.6, 0.0),
                                    (0.7, 0.0),
                                    (0.8, 0.0),
                                    ]):
            wcs = Tan(ra, dec, (W+1)/2. + dx, (H+1)/2. + dy,
                      -pixscale, 0., 0., pixscale, W, H)
            pix = np.zeros((H,W), np.float32)
            pix[H/2, :] = 1.
            pix[:, W/2] = 1.
    
            Yo,Xo,Yi,Xi,(cpix,) = resample_with_wcs(cowcs, wcs, [pix], 3)
            Yo2,Xo2,Yi2,Xi2,(pypix,) = resample_with_wcs(cowcs, wcs, [pix], 3, cinterp=False)
            cim = np.zeros((H,W))
            cim[Yo,Xo] = cpix
            pyim = np.zeros((H,W))
            pyim[Yo2,Xo2] = pypix

            plt.clf()
            plt.plot(cim[0,:], 'b-', alpha=0.5)
            plt.plot(cim[H/4,:], 'c-', alpha=0.5)
            plt.plot(pyim[0,:], 'r-', alpha=0.5)
            plt.plot(pyim[H/4,:], 'm-', alpha=0.5)
            plt.plot(1000. * (cim[0,:] - pyim[0,:]), 'k-', alpha=0.5)
            plt.savefig('p2-%02i.png' % i)
        sys.exit(0)
    
    ra,dec = 219.577111, 54.52
    pixscale = 2.75 / 3600.
    #W,H = 2048, 2048
    W,H = 512, 512
    #W,H = 100,100
    cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                -pixscale, 0., 0., pixscale, W, H)
    cowcs.write_to('co.wcs')
    
    if True:
        #intfn = '05579a167-w1-int-1b.fits'
        intfn = 'wise-frames/9a/05579a/167/05579a167-w1-int-1b.fits'
        wcs = Sip(intfn)
        pix = fitsio.read(intfn)
        pix[np.logical_not(np.isfinite(pix))] = 0.
        print 'pix', pix.shape, pix.dtype

    
    for i in range(5):
        t0 = time.clock()
        Yo,Xo,Yi,Xi,ims = resample_with_wcs(cowcs, wcs, [pix], 3)
        t1 = time.clock() - t0
        print 'C resampling took', t1

    t0 = time.clock()
    Yo2,Xo2,Yi2,Xi2,ims2 = resample_with_wcs(cowcs, wcs, [pix], 3, cinterp=False, table=False)
    t2 = time.clock() - t0
    print 'py resampling took', t2
    
    out = np.zeros((H,W))
    out[Yo,Xo] = ims[0]
    fitsio.write('resampled-c.fits', out, clobber=True)
    cout = out
    
    out = np.zeros((H,W))
    out[Yo,Xo] = ims2[0]
    fitsio.write('resampled-py.fits', out, clobber=True)
    pyout = out

    plt.clf()
    plt.imshow(cout, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.savefig('c.png')
    plt.clf()
    plt.imshow(pyout, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.savefig('py.png')

    plt.clf()
    plt.imshow(cout - pyout, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.savefig('diff.png')

    print 'Max diff:', np.abs(cout - pyout).max()

