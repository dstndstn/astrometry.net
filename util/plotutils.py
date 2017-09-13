# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches
import numpy as np
import pylab as plt
from matplotlib.ticker import FixedFormatter

import functools

class NanColormap(matplotlib.colors.Colormap):
    '''
    A Colormap that wraps another colormap, but replaces non-finite values
    with a fixed color.
    '''
    def __init__(self, cmap, nancolor):
        self.cmap = cmap
        self.nanrgba = matplotlib.colors.colorConverter.to_rgba(nancolor)
    def __call__(self, data, **kwargs):
        rgba = self.cmap(data, **kwargs)
        # 'data' is a MaskedArray, apparently...
        if np.all(data.mask == False):
            return rgba
        iy,ix = np.nonzero(data.mask)
        #print 'NanColormap: replacing', len(iy), 'pixels with', self.nanrgba
        # nanrgba are floats in [0,1]; convert to uint8 in [0,255].
        rgba[iy,ix, :] = np.clip(255. * np.array(self.nanrgba), 0, 255).astype(np.uint8)
        return rgba

    def __getattr__(self, name):
        ''' delegate to underlying colormap. '''
        return getattr(self.cmap, name)

def _imshow_better_defaults(imshowfunc, X, interpolation='nearest', origin='lower',
                            cmap='gray', ticks=True, **kwargs):
    '''
    An "imshow" wrapper that uses more sensible defaults.
    '''
    X = imshowfunc(X, interpolation=interpolation, origin=origin, cmap=cmap, **kwargs)
    if not ticks:
        plt.xticks([]); plt.yticks([])
    return X

def _imshow_nan(imshowfunc, X, nancolor='0.5', cmap=None, vmin=None, vmax=None, **kwargs):
    '''
    An "imshow" work-alike that replaces non-finite values by a fixed color.
    '''
    if np.all(np.isfinite(X)):
        return imshowfunc(X, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    # X has non-finite values.  Time to get tricky.
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap = NanColormap(cmap, nancolor)
    if vmin is None or vmax is None:
        I = np.flatnonzero(np.isfinite(X))
        if vmin is None:
            try:
                vmin = X.flat[I].min()
            except ValueError:
                vmin = 0.
        if vmax is None:
            try:
                vmax = X.flat[I].max()
            except ValueError:
                vmax = 0.
    return imshowfunc(X, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

'''
A plt.imshow() work-alike, except with defaults: interpolation='nearest', origin='lower'.
'''
imshow_better_defaults = functools.partial(_imshow_better_defaults, plt.imshow)

'''
A plt.imshow() work-alike, except handles non-finite values.  Accepts
an additional kwarg: nancolor='0.5'
'''
imshow_nan             = functools.partial(_imshow_nan, plt.imshow)

'''
My version of plt.imshow that uses imshow_better_defaults and imshow_nan.
'''
dimshow = functools.partial(_imshow_better_defaults, imshow_nan)

def replace_matplotlib_functions():
    '''
    Replaces plt.imshow with a function that handles non-finite values
    and has the defaults interpolation='nearest', origin='lower'.
    '''
    plt.imshow = dimshow




class PlotSequence(object):
    def __init__(self, basefn, format='%02i', suffix='png',
                 suffixes=None):
        self.ploti = 0
        self.basefn = basefn
        self.format = format
        if suffixes is None:
            self.suffixes = [suffix]
        else:
            self.suffixes = suffixes
        self.pattern = self.basefn + '-%s.%s'
        self.printfn = True

    def skip(self, n=1):
        self.ploti += n
    def skipto(self, n):
        self.ploti = n

    def getnextlist(self):
        #lst = ['%s-%s.%s' % (self.basefn, self.format % self.ploti, suff)
        lst = [self.pattern % (self.format % self.ploti, suff)
               for suff in self.suffixes]
        self.ploti += 1
        return lst

    def getnext(self):
        lst = self.getnextlist()
        if len(lst) == 1:
            return lst[0]
        return lst

    def savefig(self, **kwargs):
        import pylab as plt
        for fn in self.getnextlist():
            plt.savefig(fn, **kwargs)
            if self.printfn:
                print('saved', fn)

def loghist(x, y, nbins=100,
            hot=True, doclf=True, docolorbar=True, lo=0.3,
            imshowargs={},
            clampxlo=False, clampxlo_val=None, clampxlo_to=None,
            clampxhi=False, clampxhi_val=None, clampxhi_to=None,
            clampylo=False, clampylo_val=None, clampylo_to=None,
            clampyhi=False, clampyhi_val=None, clampyhi_to=None,
            clamp=None, clamp_to=None,
            **kwargs):
    #np.seterr(all='warn')
    if doclf:
        plt.clf()
    myargs = kwargs.copy()
    if not 'bins' in myargs:
        myargs['bins'] = nbins

    rng = kwargs.get('range', None)
    x = np.array(x)
    y = np.array(y)

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        K = np.flatnonzero(np.isfinite(x) * np.isfinite(y))
        print('loghist: cutting to', len(K), 'of', len(x), 'finite values')
        x = x[K]
        y = y[K]

    if clamp is True:
        clamp = rng
    if clamp is not None:
        ((clampxlo_val, clampxhi_val),(clampylo_val, clampyhi_val)) = clamp
    if clamp_to is not None:
        ((clampxlo_to, clampxhi_to),(clampylo_to, clampyhi_to)) = clamp_to
    if clampxlo:
        if clampxlo_val is None:
            if rng is None:
                raise RuntimeError('clampxlo, but no clampxlo_val or range')
            clampxlo_val = rng[0][0]
    if clampxlo_val is not None:
        if clampxlo_to is None:
            clampxlo_to = clampxlo_val
        x[x < clampxlo_val] = clampxlo_to
    if clampxhi:
        if clampxhi_val is None:
            if rng is None:
                raise RuntimeError('clampxhi, but no clampxhi_val or range')
            clampxhi_val = rng[0][1]
    if clampxhi_val is not None:
        if clampxhi_to is None:
            clampxhi_to = clampxhi_val
        x[x > clampxhi_val] = clampxhi_to
    if clampylo:
        if clampylo_val is None:
            if rng is None:
                raise RuntimeError('clampylo, but no clampylo_val or range')
            clampylo_val = rng[1][0]
    if clampylo_val is not None:
        if clampylo_to is None:
            clampylo_to = clampylo_val
        y[y < clampylo_val] = clampylo_to
    if clampyhi:
        if clampyhi_val is None:
            if rng is None:
                raise RuntimeError('clampyhi, but no clampyhi_val or range')
            clampyhi_val = rng[1][1]
    if clampyhi_val is not None:
        if clampyhi_to is None:
            clampyhi_to = clampyhi_val
        y[y > clampyhi_val] = clampyhi_to




    (H,xe,ye) = np.histogram2d(x, y, **myargs)

    L = np.log10(np.maximum(lo, H.T))
    myargs = dict(extent=(min(xe), max(xe), min(ye), max(ye)),
                  aspect='auto',
                  interpolation='nearest', origin='lower')
    myargs.update(imshowargs)
    plt.imshow(L, **myargs)
    if hot:
        plt.hot()
    if docolorbar:
        r = [np.log10(lo)] + list(range(int(np.ceil(L.max()))))
        # print 'loghist: L max', L.max(), 'r', r
        plt.colorbar(ticks=r, format=FixedFormatter(
            ['0'] + ['%i'%(10**ri) for ri in r[1:]]))
    #set_fp_err()
    return H, xe, ye

def plothist(x, y, nbins=100, log=False,
             doclf=True, docolorbar=True, dohot=True,
             plo=None, phi=None,
             scale=None,
             imshowargs={}, **hist2dargs):
    if log:
        return loghist(x, y, nbins=nbins, doclf=doclf, docolorbar=docolorbar,
                       dohot=dohot, imshowargs=imshowargs) #, **kwargs)

    if doclf:
        plt.clf()
    (H,xe,ye) = np.histogram2d(x, y, nbins, **hist2dargs)
    if scale is not None:
        H *= scale
    myargs = dict(extent=(min(xe), max(xe), min(ye), max(ye)),
                  aspect='auto',
                  interpolation='nearest', origin='lower')
    vmin = None
    if plo is not None:
        vmin = np.percentile(H.ravel(), plo)
        myargs.update(vmin=vmin)
    if phi is not None:
        vmin = imshowargs.get('vmin', vmin)
        vmax = np.percentile(H.ravel(), phi)
        if vmax != vmin:
            myargs.update(vmax=vmax)
    myargs.update(imshowargs)
    plt.imshow(H.T, **myargs)
    if dohot:
        plt.hot()
    if docolorbar:
        plt.colorbar()
    return H, xe, ye

def setRadecAxes(ramin, ramax, decmin, decmax):
    rl,rh = ramin,ramax
    dl,dh = decmin,decmax
    rascale = np.cos(np.deg2rad((dl+dh)/2.))
    ax = [rh,rl, dl,dh]
    plt.axis(ax)
    plt.gca().set_aspect(1./rascale, adjustable='box', anchor='C')
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    return ax

import matplotlib.colors as mc
class ArcsinhNormalize(mc.Normalize):
    def __init__(self, mean=None, std=None, **kwargs):
        self.mean = mean
        self.std = std
        mc.Normalize.__init__(self, **kwargs)

    def _map(self, X, out=None):
        Y = (X - self.mean) / self.std
        args = (Y,)
        if out is not None:
            args = args + (out,)
        return np.arcsinh(*args)

    def __call__(self, value, clip=None):
        # copied from Normalize since it's not easy to subclass
        if clip is None:
            clip = self.clip
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)   # Or should it be all masked?  Or 0.5?
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
            # ma division is very slow; we can take a shortcut
            resdat = result.data
            self._map(resdat, resdat)
            vmin = self._map(vmin)
            vmax = self._map(vmax)
            resdat -= vmin
            resdat /= (vmax - vmin)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result


from matplotlib.colors import LinearSegmentedColormap

# a colormap that goes from white to black: the opposite of matplotlib.gray()
antigray = LinearSegmentedColormap('antigray',
                                   {'red':   ((0., 1, 1), (1., 0, 0)),
                                    'green': ((0., 1, 1), (1., 0, 0)),
                                    'blue':  ((0., 1, 1), (1., 0, 0))})

bluegrayred = LinearSegmentedColormap('bluegrayred',
                                      {'red':   ((0., -1, 0),
                                                 (1., 1, -1)),
                                       'green': ((0., -1,   0),
                                                 (0.5,0.5, 0.5),
                                                 (1., 0, -1)),
                                       'blue':  ((0., -1, 1),
                                                 (1., 0, -1))})

# x, y0, y1
_redgreen_data =  {'red':   ((0.,  -100,  1),
                             #(0.5,  0,  0),
                             #(0.5,  0.1, 0),
                             (0.49, 0.1, 0),
                             (0.491, 0, 0),
                             (0.51,  0, 0),
                             (0.511,  0, 0.1),
                             (1.,   0, -100)),
                   'green': ((0.,  -100,  0),
                             #(0.5,  0,  0),
                             #(0.5,  0,  0.1),
                             (0.49, 0.1, 0),
                             (0.491, 0, 0),
                             (0.51,  0, 0),
                             (0.511,  0, 0.1),
                             (1.,   1, -100)),
                   'blue':  ((0.,  -100,  0),
                             (1.,   0, -100))}
redgreen = LinearSegmentedColormap('redgreen',   _redgreen_data)

def hist_ints(x, step=1, **kwargs):
    '''
    Creates a histogram of integers.  The number of bins is set to the
    range of the data (+1).  That is, each integer gets its own bin.
    '''
    kwargs['bins'] = x.max()/step - x.min()/step + 1
    kwargs['range'] = ( (x.min()/int(step))*step - 0.5,
                        ((x.max()/int(step))*step + 0.5) )
    return plt.hist(x, **kwargs)

def hist2d_with_outliers(x, y, xbins, ybins, nout):
    '''
    Creates a 2D histogram from the given data, and returns a list of
    the indices in the data of points that lie in low-occupancy cells
    (where the histogram counts is < "nout").

    The "xbins" and "ybins" arguments are passed to numpy.histogram2d.

    You probably want to show the histogram with:

      (H, outliers, xe, ye) = hist2d_with_outliers(x, y, 10, 10, 10)
      imshow(H, extent=(min(xe), max(xe), min(ye), max(ye)), aspect='auto')
      plot(x[outliers], y[outliers], 'r.')

    Returns: (H, outliers, xe, ye)

      H: 2D histogram image
      outliers: array of integer indices of the outliers
      xe: x edges chosen by histgram2d
      ye: y edges chosen by histgram2d

    '''
    # returns (density image, indices of outliers)
    (H,xe,ye) = plt.histogram2d(x, y, (xbins,ybins))
    Out = np.array([]).astype(int)
    for i in range(len(xe)-1):
        for j in range(len(ye)-1):
            if H[i,j] > nout:
                continue
            if H[i,j] == 0:
                continue
            H[i,j] = 0
            Out = np.append(Out, np.flatnonzero((x >= xe[i]) *
                                                (x <  xe[i+1]) *
                                                (y >= ye[j]) *
                                                (y <  ye[j+1])))
    return (H.T, Out, xe, ye)


# You probably want to set the keyword radius=R
def circle(xy=None, x=None, y=None, **kwargs):
    if xy is None:
        if x is None or y is None:
            raise RuntimeError('circle: need x and y')
        xy = np.array([x,y])
    c = matplotlib.patches.Circle(xy=xy, **kwargs)
    a=plt.gca()
    c.set_clip_box(a.bbox)
    a.add_artist(c)
    return c

def ellipse(xy=None, x=None, y=None, **kwargs):
    if xy is None:
        if x is None or y is None:
            raise RuntimeError('ellipse: need x and y')
        xy = np.array([x,y])
    c = matplotlib.patches.Ellipse(xy=xy, **kwargs)
    a=plt.gca()
    c.set_clip_box(a.bbox)
    a.add_artist(c)
    return c

# return (pixel width, pixel height) of the axes area.
def get_axes_pixel_size():
    dpi = plt.gcf().get_dpi()
    figsize = plt.gcf().get_size_inches()
    axpos = plt.gca().get_position()
    pixw = figsize[0] * dpi * axpos.width
    pixh = figsize[1] * dpi * axpos.height
    return (pixw, pixh)

    # test:
    if False:
        figure(dpi=100)
        (w,h) = get_axes_pixel_size()
        # not clear why this is required...
        w += 1
        h += 1
        img = zeros((h,w))
        img[:,::2] = 1.
        img[::2,:] = 1.
        imshow(img, extent=(0,w,0,h), aspect='auto', cmap=antigray)
        xlim(0,w)
        ylim(0,h)
        savefig('imtest.png')
        sys.exit(0)

# returns (x data units per pixel, y data units per pixel)
# given the current plot range, figure size, and axes position.
def get_pixel_scales():
    a = plt.axis()
    (pixw, pixh) = get_axes_pixel_size()
    return ((a[1]-a[0])/float(pixw), (a[3]-a[2])/float(pixh))

def set_image_color_percentiles(image, plo, phi):
    # hackery...
    I = image.copy().ravel()
    I.sort()
    N = len(I)
    mn = I[max(0, int(round(plo * N / 100.)))]
    mx = I[min(N-1, int(round(phi * N / 100.)))]
    plt.gci().set_clim(mn, mx)
    return (mn,mx)




if __name__ == '__main__':

    X = np.arange(25.).reshape((5,5))
    X[2:4,3:4] = np.nan
    print(X)

    plt.clf()
    imshow_nan(X, interpolation='nearest')
    plt.savefig('1.png')

    dimshow(X)
    plt.savefig('2.png')

    replace_matplotlib_functions()

    plt.clf()
    plt.imshow(X, interpolation='nearest')
    plt.savefig('3.png')

