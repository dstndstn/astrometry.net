# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from astrometry.plot.plotstuff_c import *

# Could consider using swig's "addmethods" mechanism to create this "class" rep.

class Plotstuff(object):
    format_map = {'png': PLOTSTUFF_FORMAT_PNG,
                  'jpg': PLOTSTUFF_FORMAT_JPG,
                  'ppm': PLOTSTUFF_FORMAT_PPM,
                  'pdf': PLOTSTUFF_FORMAT_PDF,
                  'fits': PLOTSTUFF_FORMAT_FITS,
                  }


    def __init__(self, outformat=None, size=None, ra=None, dec=None, width=None,
                 rdw=None, wcsfn=None, wcsext=0, alpha=1., outfn=None):
        '''
        size: (W, H), integer pixels
        '''
        p = plotstuff_new()
        self.pargs = p
        if outformat is not None:
            outformat = Plotstuff.format_map.get(outformat, outformat)
            self.outformat = outformat
        if outfn is not None:
            self.outfn = outfn
        if size is not None:
            self.size = size
            self.color = 'black'
            self.alpha = alpha
            self.plot('fill')
        if ra is not None and dec is not None and width is not None:
            self.set_wcs_box(ra, dec, width)
        if rdw is not None:
            self.set_wcs_box(*rdw)
        if wcsfn is not None:
            self.wcs_file = (wcsfn, wcsext)
            if size is None:
                plotstuff_set_size_wcs(self.pargs)

    def __del__(self):
        #print 'plotstuff.__del__, pargs=', self.pargs
        plotstuff_free(self.pargs)

    def __getattr__(self, name):
        if name == 'xy':
            return plot_xy_get(self.pargs)
        elif name == 'index':
            return plot_index_get(self.pargs)
        elif name == 'radec':
            return plot_radec_get(self.pargs)
        elif name == 'match':
            return plot_match_get(self.pargs)
        elif name == 'image':
            return plot_image_get(self.pargs)
        elif name == 'outline':
            return plot_outline_get(self.pargs)
        elif name == 'grid':
            return plot_grid_get(self.pargs)
        elif name in ['ann', 'annotations']:
            return plot_annotations_get(self.pargs)
        elif name == 'healpix':
            return plot_healpix_get(self.pargs)
        return getattr(self.pargs, name)

    def __setattr__(self, name, val):
        if name == 'pargs':
            #print 'plotstuff.py: setting pargs to', val
            self.__dict__[name] = val
        elif name == 'size':
            #print 'plotstuff.py: setting plot size of', self.pargs, 'to %i,%i' % (val[0], val[1])
            plotstuff_set_size(self.pargs, val[0], val[1])
        elif name == 'color':
            #print 'plotstuff.py: setting color to "%s"' % val
            self.set_color(val)
        elif name == 'rgb':
            plotstuff_set_rgba2(self.pargs, val[0], val[1], val[2],
                           plotstuff_get_alpha(self.pargs))
        elif name == 'bg_rgba':
            plotstuff_set_bgrgba2(self.pargs, val[0], val[1], val[2], val[3])
        elif name == 'bg_box':
            self.pargs.bg_box = val
        elif name == 'rgba':
            plotstuff_set_rgba2(self.pargs, val[0], val[1], val[2], val[3])
        elif name == 'alpha':
            self.set_alpha(val)
        elif name == 'lw':
            self.pargs.lw = float(val)
        elif name == 'marker' and type(val) is str:
            plotstuff_set_marker(self.pargs, val)
        elif name == 'markersize':
            plotstuff_set_markersize(self.pargs, val)
        elif name == 'wcs_file':
            if type(val) is tuple:
                plotstuff_set_wcs_file(self.pargs, *val)
            else:
                plotstuff_set_wcs_file(self.pargs, val, 0)
        elif name == 'wcs':
            plotstuff_set_wcs(self.pargs, val)
        elif name == 'wcs_tan':
            plotstuff_set_wcs_tan(self.pargs, val)
        elif name == 'wcs_sip':
            plotstuff_set_wcs_sip(self.pargs, val)
        elif name == 'text_bg_alpha':
            plotstuff_set_text_bg_alpha(self.pargs, val)
        #elif name == 'operator':
        #    print 'val:', val
        #    self.pargs.op = val
        else:
            self.pargs.__setattr__(name, val)

    def line_constant_ra(self, ra, declo, dechi, startwithmove=True):
        return plotstuff_line_constant_ra(self.pargs, ra, declo, dechi, startwithmove)
    def line_constant_dec(self, dec, ralo, rahi):
        return plotstuff_line_constant_dec(self.pargs, dec, ralo, rahi)
    def line_constant_dec2(self, dec, ralo, rahi, rastep):
        return plotstuff_line_constant_dec2(self.pargs, dec, ralo, rahi, rastep)
    def fill(self):
        return plotstuff_fill(self.pargs)
        #return self.plot('fill')
    def stroke(self):
        return plotstuff_stroke(self.pargs)
    def fill_preserve(self):
        return plotstuff_fill_preserve(self.pargs)
    def stroke_preserve(self):
        return plotstuff_stroke_preserve(self.pargs)
    def move_to_radec(self, ra, dec):
        return plotstuff_move_to_radec(self.pargs, ra, dec)
    def line_to_radec(self, ra, dec):
        return plotstuff_line_to_radec(self.pargs, ra, dec)
    def move_to_xy(self, x, y):
        plotstuff_move_to(self.pargs, x, y)
    def line_to_xy(self, x, y):
        plotstuff_line_to(self.pargs, x, y)
    def close_path(self):
        return plotstuff_close_path(self.pargs)
    def dashed(self, dashlen):
        plotstuff_set_dashed(self.pargs, dashlen)
    def solid(self):
        plotstuff_set_solid(self.pargs)

    def set_size_from_wcs(self):
        self.pargs.set_size_from_wcs()

    def polygon(self, xy, makeConvex=True):
        import numpy as np
        if makeConvex:
            cx = sum(x for x,y in xy) / float(len(xy))
            cy = sum(y for x,y in xy) / float(len(xy))
            angles = np.array([np.arctan2(y - cy, x - cx)
                               for x,y in xy])
            I = np.argsort(angles)
        else:
            I = np.arange(len(xy))

        for j,i in enumerate(I):
            x,y = xy[i]
            if j == 0:
                self.move_to_xy(x, y)
            else:
                self.line_to_xy(x, y)

    def get_image_as_numpy(self, flip=False, out=None):
        # Caution: possible memory-handling problem with using "out"
        return self.pargs.get_image_as_numpy(flip, out)

    def get_image_as_numpy_view(self):
        return self.pargs.get_image_as_numpy_view()

    def set_image_from_numpy(self, img, flip=False):
        self.pargs.set_image_from_numpy(img, flip)

    def view_image_as_numpy(self):
        return self.pargs.view_image_as_numpy()

    def apply_settings(self):
        plotstuff_builtin_apply(self.pargs.cairo, self.pargs)

    def plot(self, layer):
        return plotstuff_plot_layer(self.pargs, layer)

    def scale_wcs(self, scale):
        plotstuff_scale_wcs(self.pargs, scale)

    def rotate_wcs(self, angle):
        plotstuff_rotate_wcs(self.pargs, angle)

    def set_wcs_box(self, ra, dec, width):
        plotstuff_set_wcs_box(self.pargs, ra, dec, width)

    def set_color(self, color):
        #print 'calling plotstuff_set_color(., \"%s\")' % color
        x = plotstuff_set_color(self.pargs, color)
        return x

    def set_alpha(self, a):
        x = plotstuff_set_alpha(self.pargs, a)

    def plot_grid(self, rastep, decstep, ralabelstep=None, declabelstep=None):
        import numpy as np

        grid = plot_grid_get(self.pargs)
        grid.rastep = rastep
        grid.decstep = decstep
        rformat = None
        if ralabelstep is None:
            ralabelstep = 0
        else:
            rdigits = max(0, np.ceil(-np.log10(ralabelstep)))
            rformat = '%.' + '%i'%rdigits + 'f'
        dformat = None
        if declabelstep is None:
            declabelstep = 0
        else:
            ddigits = max(0, np.ceil(-np.log10(declabelstep)))
            dformat = '%.' + '%i'%ddigits + 'f'
        if rformat is not None or dformat is not None:
            rformat = rformat or '%.2f'
            dformat = dformat or '%.2f'
            grid.set_formats(rformat, dformat)

        grid.ralabelstep = ralabelstep
        grid.declabelstep = declabelstep
        self.plot('grid')

    def clear(self):
        plotstuff_clear(self.pargs)

    def write(self, filename=None):
        if filename is not None:
            self.outfn = filename
        plotstuff_output(self.pargs)

    def text_xy(self, x, y, text):
        plotstuff_text_xy(self.pargs, x, y, text)

    def text_radec(self, ra, dec, text):
        plotstuff_text_radec(self.pargs, ra, dec, text)

    def stack_marker(self, x, y):
        plotstuff_stack_marker(self.pargs, x, y)

    def marker_xy(self, x,y):
        rtn = plotstuff_marker(self.pargs, x,y)

    def marker_radec(self, ra, dec):
        #print 'marker_radec', ra, dec
        rtn = plotstuff_marker_radec(self.pargs, ra, dec)
        #print '-> ', rtn
    def set_markersize(self, size):
        plotstuff_set_markersize(self.pargs, size)

    def plot_stack(self):
        plotstuff_plot_stack(self.pargs, self.pargs.cairo)
