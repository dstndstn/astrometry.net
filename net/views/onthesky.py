import os

from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net import settings

def plot_aitoff_wcs_outline(wcsfn, plotfn, width=400):
    #from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    anutil = ps

    ps.log_init(3)

    height = width/2
    # Create Hammer-Aitoff WCS of the appropriate size.
    wcs = anutil.anwcs_create_allsky_hammer_aitoff(0., 0., width, height)

    plot = ps.Plotstuff(outformat='png', size=(width, height))
    plot.wcs = wcs
    plot.color = 'white'
    out = plot.outline
    #out.fill = 1
    out.wcs_file = wcsfn
    anutil.anwcs_print_stdout(out.wcs)
    plot.plot('outline')

    plot.color = 'gray'
    plot.plot_grid(60, 30, 60, 30)

    plot.write(plotfn)
    
    
