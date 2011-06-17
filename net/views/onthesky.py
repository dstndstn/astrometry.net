import os

from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net import settings

def plot_aitoff_wcs_outline(wcsfn, plotfn, width=400):
    #from astrometry.util import util as anutil
    from astrometry.blind import plotstuff as ps
    anutil = ps

    ps.log_init(2)

    height = width/2
    # Create Hammer-Aitoff WCS of the appropriate size.
    wcs = anutil.anwcs_create_allsky_hammer_aitoff(0., 0., width, height)

    plot = ps.Plotstuff(outformat='png', size=(width, height))
    plot.wcs = wcs

    #plot.plot_grid(60, 30, 60, 30)
    plot.fontsize = 12
    ras = [-180, -120, -60, 0, 60, 120, 180]
    decs = [-60, -30, 0, 30, 60]
    plot.rgb = (0.3,0.3,0.3)
    plot.apply_settings()
    for ra in ras:
        ps.plotstuff_line_constant_ra(plot.pargs, ra, -90, 90)
        ps.plotstuff_stroke(plot.pargs)
    for dec in decs:
        ps.plotstuff_line_constant_dec(plot.pargs, dec, -180, 180)
        ps.plotstuff_stroke(plot.pargs)
    plot.color = 'gray'
    for ra in ras:
        ps.plotstuff_move_to_radec(plot.pargs, ra, 0)
        ps.plotstuff_text_radec(plot.pargs, ra, 0, '%i'%((ra+360)%360))
        ps.plotstuff_stroke(plot.pargs)
    for dec in decs:
        ps.plotstuff_move_to_radec(plot.pargs, 0, dec)
        ps.plotstuff_text_radec(plot.pargs, 0, dec, '%i'%dec)
        ps.plotstuff_stroke(plot.pargs)
    
    plot.color = 'white'
    plot.lw = 3
    out = plot.outline
    #out.fill = 1
    out.wcs_file = wcsfn
    anutil.anwcs_print_stdout(out.wcs)
    plot.plot('outline')

    plot.write(plotfn)
    
    
