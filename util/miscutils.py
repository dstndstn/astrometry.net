# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from math import pi
import numpy as np

def estimate_mode(img, lo=None, hi=None, plo=1, phi=70, bins1=30,
                   flo=0.5, fhi=0.8, bins2=30,
                   return_fit=False, raiseOnWarn=False):
    # Estimate sky level by: compute the histogram within [lo,hi], fit
    # a parabola to the log-counts, return the argmax of that parabola.

    # Coarse bin to find the peak (mode)
    if lo is None:
        lo = np.percentile(img, plo)
    if hi is None:
        hi = np.percentile(img, phi)

    binedges1 = np.linspace(lo, hi, bins1+1)
    counts1,e = np.histogram(img.ravel(), bins=binedges1)
    bincenters1 = binedges1[:-1] + (binedges1[1]-binedges1[0])/2.
    maxbin = np.argmax(counts1)
    maxcount = counts1[maxbin]
    mode = bincenters1[maxbin]

    # Search for bin containing < {flo,fhi} * maxcount
    ilo = maxbin
    while ilo > 0:
        ilo -= 1
        if counts1[ilo] < flo*maxcount:
            break
    ihi = maxbin
    while ihi < bins1-1:
        ihi += 1
        if counts1[ihi] < fhi*maxcount:
            break

    lo = bincenters1[ilo]
    hi = bincenters1[ihi]

    binedges = np.linspace(lo, hi, bins2)
    counts,e = np.histogram(img.ravel(), bins=binedges)
    bincenters = binedges[:-1] + (binedges[1]-binedges[0])/2.

    b = np.log10(np.maximum(1, counts))

    xscale = 0.5 * (hi - lo)
    x0 = (hi + lo) / 2.
    x = (bincenters - x0) / xscale

    A = np.zeros((len(x), 3))
    A[:,0] = 1.
    A[:,1] = x
    A[:,2] = x**2
    res = np.linalg.lstsq(A, b, rcond=None)
    X = res[0]
    mx = -X[1] / (2. * X[2])
    mx = (mx * xscale) + x0

    warn = None

    if not (mx > lo and mx < hi):
        if raiseOnWarn:
            raise ValueError('sky estimate not bracketed by peak: lo %f, sky %f, hi %f' % (lo, mx, hi))
        warn = 'WARNING: sky estimate not bracketed by peak: lo %f, sky %f, hi %f' % (lo, mx, hi)

    if return_fit:
        bfit = X[0] + X[1] * x + X[2] * x**2
        return (x * xscale + x0, b, bfit, mx, warn, binedges1,counts1)

    return mx


def parse_ranges(s):
    '''
    Parse PBS job array-style ranges: NNN,MMM-NNN,PPP

    *s*: string

    Returns: [ int, int, ... ]
    '''
    tiles = []
    words = s.split()
    for w in words:
        for a in w.split(','):
            if '-' in a:
                aa = a.split('-')
                if len(aa) != 2:
                    raise RuntimeError('With an arg containing a dash, expect two parts, in word "%s"' % a)
                start = int(aa[0])
                end = int(aa[1])
                for i in range(start, end+1):
                    tiles.append(i)
            else:
                tiles.append(int(a))
    return tiles


def patch_image(img, mask, dxdy = [(-1,0),(1,0),(0,-1),(0,1)],
                required=None):
    '''
    Patch masked pixels by iteratively averaging non-masked neighboring pixels.

    WARNING: this modifies BOTH the "img" and "mask" arrays!

    mask: True for good pixels
    required: if non-None: True for pixels you want to be patched.
    dxdy: Pixels to average in, relative to pixels to be patched.

    Returns True if patching was successful.
    '''
    assert(img.shape == mask.shape)
    assert(len(img.shape) == 2)
    h,w = img.shape
    Nlast = -1
    while True:
        needpatching = np.logical_not(mask)
        if required is not None:
            needpatching *= required
        I = np.flatnonzero(needpatching)
        if len(I) == 0:
            break
        if len(I) == Nlast:
            return False
        #print 'Patching', len(I), 'pixels'
        Nlast = len(I)
        iy,ix = np.unravel_index(I, img.shape)
        psum = np.zeros(len(I), img.dtype)
        pn = np.zeros(len(I), int)

        for dx,dy in dxdy:
            ok = True
            if dx < 0:
                ok = ok * (ix >= (-dx))
            if dx > 0:
                ok = ok * (ix <= (w-1-dx))
            if dy < 0:
                ok = ok * (iy >= (-dy))
            if dy > 0:
                ok = ok * (iy <= (h-1-dy))

            # darn, NaN * False = NaN, not zero.
            finite = np.isfinite(img [iy[ok]+dy, ix[ok]+dx])
            ok[ok] *= finite

            psum[ok] += (img [iy[ok]+dy, ix[ok]+dx] *
                         mask[iy[ok]+dy, ix[ok]+dx])
            pn[ok] += mask[iy[ok]+dy, ix[ok]+dx]

            # print 'ix', ix
            # print 'iy', iy
            # print 'dx,dy', dx,dy
            # print 'ok', ok
            # print 'psum', psum
            # print 'pn', pn

        img.flat[I] = (psum / np.maximum(pn, 1)).astype(img.dtype)
        mask.flat[I] = (pn > 0)
        #print 'Patched', np.sum(pn > 0)
    return True

def clip_wcs(wcs1, wcs2, makeConvex=True, pix1=None, pix2=None):
    '''
    Returns a pixel-space polygon in WCS1 after it is clipped by WCS2.
    Returns an empty list if the two WCS headers do not intersect.

    Note that due to weakness in the clip_polygon method, wcs2 must be convex.
    If makeConvex=True, we find the convex hull and clip with that.

    If pix1 is not None, pix1=(xx,yy), xx and yy lists of pixel coordinates defining
    the boundary of the image, in CLOCKWISE order; default is the edges and midpoints.

    '''
    if pix1 is None:
        w1,h1 = wcs1.get_width(), wcs1.get_height()
        x1,x2,x3 = 0.5, w1/2., w1+0.5
        y1,y2,y3 = 0.5, h1/2., h1+0.5
        xx = [x1, x1, x1, x2, x3, x3, x3, x2]
        yy = [y1, y2, y3, y3, y3, y2, y1, y1]
    else:
        xx,yy = pix1
    #rr,dd = wcs1.pixelxy2radec(xx, yy)

    if pix2 is None:
        w2,h2 = wcs2.get_width(), wcs2.get_height()
        x1,x2,x3 = 0.5, w2/2., w2+0.5
        y1,y2,y3 = 0.5, h2/2., h2+0.5
        XX = [x1, x1, x1, x2, x3, x3, x3, x2]
        YY = [y1, y2, y3, y3, y3, y2, y1, y1]
    else:
        XX,YY = pix2
    rr,dd = wcs2.pixelxy2radec(XX, YY)
    ok,XX,YY = wcs1.radec2pixelxy(rr, dd)
    # XX,YY is the clip polygon in wcs1 pixel coords.
    # Not necessarily clockwise at this point!

    #print 'XX,YY', XX, YY

    if makeConvex:
        from scipy.spatial import ConvexHull
        points = np.vstack((XX,YY)).T
        #print 'points', points.shape
        hull = ConvexHull(points)
        #print 'Convex hull:', hull
        #print hull.vertices

        # ConvexHull returns the vertices in COUNTER-clockwise order.  Reverse.
        v = np.array(list(reversed(hull.vertices))).astype(int)
        XX = XX[v]
        YY = YY[v]

        #plt.plot(XX, YY, 'm-')
        #plt.plot(XX[0], YY[0], 'mo')
        #plt.savefig('clip2.png')
    else:
        # Ensure points are listed in CLOCKWISE order.
        crosses = []
        for i in range(len(XX)):
            j = (i + 1) % len(XX)
            k = (i + 2) % len(XX)
            xi,yi = XX[i], YY[i]
            xj,yj = XX[j], YY[j]
            xk,yk = XX[k], YY[k]
            dx1, dy1 = xj - xi, yj - yi
            dx2, dy2 = xk - xj, yk - yj
            cross = dx1 * dy2 - dy1 * dx2
            #print 'cross', cross
            crosses.append(cross)
        crosses = np.array(crosses)
        #print 'cross products', crosses
        if np.all(crosses >= 0):
            # Reverse
            #print 'Reversing wcs2 points'
            XX = np.array(list(reversed(XX)))
            YY = np.array(list(reversed(YY)))

    clip = clip_polygon(list(zip(xx, yy)), list(zip(XX, YY)))
    clip = np.array(clip)

    if False:
        import pylab as plt
        plt.clf()
        plt.plot(xx, yy, 'b.-')
        plt.plot(xx[0], yy[0], 'bo')
        plt.plot(XX, YY, 'r.-')
        plt.plot(XX[0], YY[0], 'ro')
        if len(clip) > 0:
            plt.plot(clip[:,0], clip[:,1], 'm.-', alpha=0.5, lw=2)
        plt.savefig('clip1.png')

    return clip



def polygon_area(poly):
    '''
    NOTE, unlike many of the other methods in this module, takes:

    poly = (xx,yy)

    where xx,yy MUST repeat the starting point at the end of the polygon.
    '''
    xx,yy = poly
    x,y = np.mean(xx), np.mean(yy)
    area = 0.
    for dx0,dy0,dx1,dy1 in zip(xx-x, yy-y, xx[1:]-x, yy[1:]-y):
        # area: 1/2 cross product
        area += np.abs(dx0 * dy1 - dx1 * dy0)
    return 0.5 * area

def clip_polygon(poly1, poly2):
    '''
    Returns a new polygon resulting from taking poly1 and clipping it
    to lie inside poly2.

    WARNING, the polygons must be listed in CLOCKWISE order.

    WARNING, the clipping polygon, poly2, must be CONVEX.
    '''
    # from clipper import Clipper, Point, PolyType, ClipType, PolyFillType
    # '''
    # '''
    # c = Clipper()
    # p1 = [Point(x,y) for x,y in poly1]
    # p2 = [Point(x,y) for x,y in poly2]
    # c.AddPolygon(p1, PolyType.Subject)
    # c.AddPolygon(p2, PolyType.Clip)
    # solution = []
    # pft = PolyFillType.EvenOdd
    # result = c.Execute(ClipType.Intersection, solution, pft, pft)
    # if len(solution) > 1:
    #     raise RuntimeError('Polygon clipping results in non-simple polygon')
    # assert(result)
    # #print 'Result:', result
    # #print 'Solution:', solution
    # return [(s.x, s.y) for s in solution[0]]

    # Sutherland-Hodgman algorithm -- thanks, Wikipedia!
    N2 = len(poly2)
    # clip by each edge in turn.
    for j in range(N2):
        # target "left_right" value
        clip1 = poly2[j]
        clip2 = poly2[(j+1)%N2]
        LRinside = _left_right(clip1, clip2, poly2[(j+2)%N2])
        # are poly vertices inside or outside the clip polygon?
        isinside = [_left_right(clip1, clip2, p) == LRinside
                    for p in poly1]
        # the resulting clipped polygon
        clipped = []
        N1 = len(poly1)
        for i in range(N1):
            S = poly1[i]
            E = poly1[(i+1)%N1]
            Sin = isinside[i]
            Ein = isinside[(i+1)%N1]
            if Ein:
                if not Sin:
                    clipped.append(line_intersection(clip1, clip2, S, E))
                clipped.append(E)
            else:
                if Sin:
                    clipped.append(line_intersection(clip1, clip2, S, E))
        poly1 = clipped
    return poly1


def polygons_intersect(poly1, poly2):
    '''
    Determines whether the given 2-D polygons intersect.

    poly1, poly2: np arrays with shape (N,2)
    '''

    # Check whether any points in poly1 are inside poly2,
    # or vice versa.
    for (px,py) in poly1:
        if point_in_poly(px,py, poly2):
            return (px,py)
    for (px,py) in poly2:
        if point_in_poly(px,py, poly1):
            return (px,py)

    # Check for intersections between line segments.  O(n^2) brutish
    N1 = len(poly1)
    N2 = len(poly2)

    for i in range(N1):
        for j in range(N2):
            xy = line_segments_intersect(poly1[i % N1, :], poly1[(i+1) % N1, :],
                                         poly2[j % N2, :], poly2[(j+1) % N2, :])
            if xy:
                return xy
    return False


def line_segments_intersect(xy1, xy2, xy3, xy4):
    '''
    Determines whether the two given line segments intersect;

    (x1,y1) to (x2,y2)
    and
    (x3,y3) to (x4,y4)
    '''
    (x1,y1) = xy1
    (x2,y2) = xy2
    (x3,y3) = xy3
    (x4,y4) = xy4
    x,y = line_intersection((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    if x is None:
        # Parallel lines
        return False
    if x1 == x2:
        p1,p2 = y1,y2
        p = y
    else:
        p1,p2 = x1,x2
        p = x

    if not ((p >= min(p1,p2)) and (p <= max(p1,p2))):
        return False

    if x3 == x4:
        p1,p2 = y3,y4
        p = y
    else:
        p1,p2 = x3,x4
        p = x

    if not ((p >= min(p1,p2)) and (p <= max(p1,p2))):
        return False
    return (x,y)


def line_intersection(xy1, xy2, xy3, xy4):
    '''
    Determines the point where the lines described by
    (x1,y1) to (x2,y2)
    and
    (x3,y3) to (x4,y4)
    intersect.

    Note that this may be beyond the endpoints of the line segments.

    Probably raises an exception if the lines are parallel, or does
    something numerically crazy.
    '''
    (x1,y1) = xy1
    (x2,y2) = xy2
    (x3,y3) = xy3
    (x4,y4) = xy4
    # This code started with the equation from Wikipedia,
    # then I added special-case handling.
    # bottom = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    # if bottom == 0:
    #   raise RuntimeError("divide by zero")
    # t1 = (x1 * y2 - y1 * x2)
    # t2 = (x3 * y4 - y3 * x4)
    # px = (t1 * (x3 - x4) - t2 * (x1 - x2)) / bottom
    # py = (t1 * (y3 - y4) - t2 * (y1 - y2)) / bottom

    # From http://wiki.processing.org/w/Line-Line_intersection
    bx = float(x2) - float(x1)
    by = float(y2) - float(y1)
    dx = float(x4) - float(x3)
    dy = float(y4) - float(y3)
    b_dot_d_perp = bx*dy - by*dx
    if b_dot_d_perp == 0:
        return None,None
    cx = float(x3) - float(x1)
    cy = float(y3) - float(y1)
    t = (cx*dy - cy*dx) / b_dot_d_perp
    return x1 + t*bx, y1 + t*by

def _left_right(xy1, xy2, xy3):
    '''
    is (x3,y3) to the 'left' or 'right' of the line from (x1,y1) to (x2,y2) ?
    '''
    (x1,y1) = xy1
    (x2,y2) = xy2
    (x3,y3) = xy3
    dx2,dy2 = x2-x1, y2-y1
    dx3,dy3 = x3-x1, y3-y1
    return (dx2 * dy3 - dx3 * dy2) > 0


def point_in_poly(x, y, poly):
    '''
    Performs a point-in-polygon test for numpy arrays of *x* and *y*
    values, and a polygon described as 2-d numpy array (with shape (N,2))

    poly: N x 2 array

    Returns a numpy array of bools.
    '''
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    inside = np.zeros(x.shape, bool)
    # This does a winding test -- count how many times a horizontal ray
    # from (-inf,y) to (x,y) crosses the boundary.
    for i in range(len(poly)):
        j = (i-1 + len(poly)) % len(poly)
        xi,xj = poly[i,0], poly[j,0]
        yi,yj = poly[i,1], poly[j,1]

        if yi == yj:
            continue

        I = np.logical_and(
            np.logical_or(np.logical_and(yi <= y, y < yj),
                          np.logical_and(yj <= y, y < yi)),
            x < (xi + ((xj - xi) * (y - yi) / (yj - yi))))
        inside[I] = np.logical_not(inside[I])
    return inside

def lanczos_filter(order, x, out=None):
    x = np.atleast_1d(x)
    nz = np.logical_and(x != 0., np.logical_and(x < order, x > -order))
    nz = np.flatnonzero(nz)
    if out is None:
        out = np.zeros(x.shape, dtype=float)
    else:
        out[x <= -order] = 0.
        out[x >=  order] = 0.
    pinz = pi * x.flat[nz]
    out.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    return out

def get_overlapping_region(xlo, xhi, xmin, xmax):
    '''
    Given a range of integer coordinates that you want to, eg, cut out
    of an image, [xlo, xhi], and bounds for the image [xmin, xmax],
    returns the range of coordinates that are in-bounds, and the
    corresponding region within the desired cutout.

    For example, say you have an image of shape H,W and you want to
    cut out a region of halfsize "hs" around pixel coordinate x,y, but
    so that coordinate x,y is centered in the cutout even if x,y is
    close to the edge.  You can do:

    cutout = np.zeros((hs*2+1, hs*2+1), img.dtype)
    iny,outy = get_overlapping_region(y-hs, y+hs, 0, H-1)
    inx,outx = get_overlapping_region(x-hs, x+hs, 0, W-1)
    cutout[outy,outx] = img[iny,inx]
    
    '''
    if xlo > xmax or xhi < xmin or xlo > xhi or xmin > xmax:
        return ([], [])

    assert(xlo <= xhi)
    assert(xmin <= xmax)

    xloclamp = max(xlo, xmin)
    Xlo = xloclamp - xlo

    xhiclamp = min(xhi, xmax)
    Xhi = Xlo + (xhiclamp - xloclamp)

    #print 'xlo, xloclamp, xhiclamp, xhi', xlo, xloclamp, xhiclamp, xhi
    assert(xloclamp >= xlo)
    assert(xloclamp >= xmin)
    assert(xloclamp <= xmax)
    assert(xhiclamp <= xhi)
    assert(xhiclamp >= xmin)
    assert(xhiclamp <= xmax)
    #print 'Xlo, Xhi, (xmax-xmin)', Xlo, Xhi, xmax-xmin
    assert(Xlo >= 0)
    assert(Xhi >= 0)
    assert(Xlo <= (xhi-xlo))
    assert(Xhi <= (xhi-xlo))

    return (slice(xloclamp, xhiclamp+1), slice(Xlo, Xhi+1))



if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    import numpy as np
    from astrometry.util.plotutils import *
    ps = PlotSequence('miscutils')

    np.random.seed(42)

    if True:
        p2 = np.array([[0,0],[0,4],[4,4],[4,0]])

        for i,p1 in enumerate([ np.array([[0,0],[0,2],[2,2],[2,0]]),
                                np.array([[-1,-1],[0,2],[2,2],[2,0]]),
                                np.array([[4,0],[0,4],[-4,0],[0,-4]]),
                                np.array([[-1,2],[2,5],[5,2],[2,-1]]),
                                ] + [None]*10):
            if p1 is None:
                p1 = np.random.uniform(high=6., low=-2, size=(4,2))
            pc = np.array(clip_polygon(p1, p2))
            plt.clf()
            I = np.array([0,1,2,3,0])
            plt.plot(p1[I,0], p1[I,1], 'b-', lw=3, alpha=0.5)
            plt.plot(p2[I,0], p2[I,1], 'k-')
            I = np.array(range(len(pc)) + [0])
            plt.plot(pc[I,0], pc[I,1], 'r-')
            plt.axis([-1,5,-1,5])
            plt.savefig('clip-%02i.png' % i)
        import sys
        sys.exit(0)

    if True:
        for i in range(20):
            if i <= 10:
                xy1 = np.array([[0,0],[0,4],[4,4],[4,0]])
            else:
                xy1 = np.random.uniform(high=10., size=(4,2))
            xy2 = np.random.uniform(high=10., size=(4,2))
            plt.clf()
            I = np.array([0,1,2,3,0])
            xy = polygons_intersect(xy1, xy2)
            if xy:
                cc = 'r'
                x,y = xy
                plt.plot(x,y, 'mo', mec='m', mfc='none', ms=20, mew=3, zorder=30)
            else:
                cc = 'k'
            plt.plot(xy1[I,0], xy1[I,1], '-', color=cc, zorder=20, lw=3)
            plt.plot(xy2[I,0], xy2[I,1], '-', color=cc, zorder=20, lw=3)
            ax = plt.axis()
            plt.axis([ax[0]-0.5, ax[1]+0.5, ax[2]-0.5, ax[3]+0.5])
            ps.savefig()

    if False:
        X,Y = np.meshgrid(np.linspace(-1,11, 20), np.linspace(-1,11, 23))
        X = X.ravel()
        Y = Y.ravel()
        for i in range(20):
            if i == 0:
                xy = np.array([[0,0],[0,10],[10,10],[10,0]])
            else:
                xy = np.random.uniform(high=10., size=(4,2))
            plt.clf()
            I = np.array([0,1,2,3,0])
            plt.plot(xy[I,0], xy[I,1], 'r-', zorder=20, lw=3)
            inside = point_in_poly(X, Y, xy)
            plt.plot(X[inside], Y[inside], 'bo')
            out = np.logical_not(inside)
            plt.plot(X[out], Y[out], 'ro')
            ax = plt.axis()
            plt.axis([ax[0]-0.5, ax[1]+0.5, ax[2]-0.5, ax[3]+0.5])
            ps.savefig()



    if True:
        # intersection()
        for i in range(20):
            if i == 0:
                x1 = x2 = 0
                y1 = 0
                y2 = 1
                x3 = 1
                x4 = -1
                y3 = 0
                y4 = 1
            elif i == 1:
                x1,y1 = 0,0
                x2,y2 = 0,1
                x3,y3 = -3,0
                x4,y4 = -2,0
            elif i == 2:
                x1,y1 = 1,0
                x2,y2 = 0,1
                x3,y3 = -3,0
                x4,y4 = -2,0
            elif i == 3:
                x1,y1 = 0,1
                x2,y2 = 1,0
                x3,y3 = 0,-3
                x4,y4 = 0,-2
            elif i == 4:
                x1,y1 = 0,0
                x2,y2 = 0,1
                x3,y3 = 0,2
                x4,y4 = 0,3
            elif i == 5:
                x1,y1 = -1,0
                x2,y2 = 1, 0
                x3,y3 = 0, 2
                x4,y4 = 0.5, 1
            else:
                xy = np.random.uniform(high=10., size=(8,))
                x1,y1,x2,y2,x3,y3,x4,y4 = xy
            plt.clf()
            plt.plot([x1,x2],[y1,y2], 'r-', zorder=20, lw=3)
            plt.plot([x3,x4],[y3,y4], 'b-', zorder=20, lw=3)
            x,y = line_intersection((x1,y1),(x2,y2),(x3,y3),(x4,y4))
            plt.plot(x, y, 'kx', ms=20, zorder=25)
            plt.plot([x1,x],[y1,y], 'k--', alpha=0.5, zorder=15)
            plt.plot([x2,x],[y2,y], 'k--', alpha=0.5, zorder=15)
            plt.plot([x3,x],[y3,y], 'k--', alpha=0.5, zorder=15)
            plt.plot([x4,x],[y4,y], 'k--', alpha=0.5, zorder=15)

            # line_segments_intersect()
            if line_segments_intersect((x1,y1),(x2,y2),(x3,y3),(x4,y4)):
                plt.plot(x,y, 'mo', mec='m', mfc='none', ms=20, mew=3, zorder=30)
            ax = plt.axis()
            plt.axis([ax[0]-0.5, ax[1]+0.5, ax[2]-0.5, ax[3]+0.5])
            ps.savefig()
