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

    plot.linestep = 1.

    plot.color = 'verydarkblue'
    plot.apply_settings()
    plot.line_constant_ra(180, -90, 90)
    plot.line_constant_ra(-180, 90, -90)
    plot.fill()

    #plot.plot_grid(60, 30, 60, 30)
    plot.fontsize = 12
    ras = [-180, -120, -60, 0, 60, 120, 180]
    decs = [-60, -30, 0, 30, 60]
    # dark gray
    plot.rgb = (0.3,0.3,0.3)
    plot.apply_settings()
    for ra in ras:
        plot.line_constant_ra(ra, -90, 90)
        plot.stroke()
    for dec in decs:
        plot.line_constant_dec(dec, -180, 180)
        plot.stroke()

    plot.color = 'gray'
    plot.apply_settings()
    for ra in ras:
        plot.move_to_radec(ra, 0)
        plot.text_radec(ra, 0, '%i'%((ra+360)%360))
        plot.stroke()
    for dec in decs:
        if dec != 0:
            plot.move_to_radec(0, dec)
            plot.text_radec(0, dec, '%+i'%dec)
            plot.stroke()
    
    plot.color = 'white'
    plot.lw = 3
    out = plot.outline
    #out.fill = 1
    out.wcs_file = wcsfn
    anutil.anwcs_print_stdout(out.wcs)
    plot.plot('outline')

    ann = plot.annotations
    ann.NGC = ann.bright = ann.HD = 0
    ann.constellations = 1
    plot.plot('annotations')

    plot.write(plotfn)
    
    
