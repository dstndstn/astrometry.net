#! /usr/bin/env python3
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import print_function
import os
import sys
import fnmatch
from astrometry.plot.plotstuff_c import *
from numpy import *
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser(usage='%prog [options] <ra> <dec> <width-in-deg> <index file> <output-filename>')
    (opt, args) = parser.parse_args()
    if len(args) != 5:
        parser.print_help()
        print()
        print('Got wrong number of arguments:', args)
        sys.exit(-1)

    ra = float(args[0])
    dec = float(args[1])
    width = float(args[2])
    indexfn = args[3]
    outfn = args[4]

    pargs = plotstuff_new()
    pargs.outformat = PLOTSTUFF_FORMAT_PNG
    pargs.outfn = outfn

    plotstuff_set_size(pargs, 800, 800)
    plotstuff_set_wcs_box(pargs, ra, dec, width)

    plotstuff_set_color(pargs, 'verydarkblue')
    plotstuff_run_command(pargs, 'fill')

    plotstuff_set_color(pargs, 'gray')
    plotstuff_set_alpha(pargs, 0.25)
    grid = plot_grid_get(pargs)
    grid.rastep = grid.decstep = 1
    grid.ralabelstep = grid.declabelstep = 0
    plotstuff_run_command(pargs, 'grid')
    plotstuff_set_color(pargs, 'gray')
    grid.rastep = grid.decstep = 0
    grid.ralabelstep = grid.declabelstep = 1
    plotstuff_set_bgrgba2(pargs, 0, 0, 0, 0.5)
    plotstuff_run_command(pargs, 'grid')

    plotstuff_set_color(pargs, 'green')
    pargs.markersize = 1
    index = plot_index_get(pargs)
    plot_index_add_file(index, indexfn)
    index.stars = 1
    index.quads = 1
    plotstuff_run_command(pargs, 'index')

    plotstuff_output(pargs)

    plotstuff_free(pargs)
    
