import os

from astrometry.net.log import *
from astrometry.net.tmpfile import *
from astrometry.net import settings

from astrometry.util import util as anutil
from astrometry.blind import plotstuff as ps

def plot_wcs_outline(wcsfn, plotfn, W=400, H=400, width=36, zoom=True,
                     zoomwidth=3.6, grid=10, hd=False):
    anutil.log_init(3)
    #anutil.log_set_level(3)

    wcs = anutil.Tan(wcsfn, 0)
    ra,dec = wcs.radec_center()

    plot = ps.Plotstuff(outformat='png', size=(W, H), rdw=(ra,dec,width))
    plot.linestep = 1.
    plot.color = 'verydarkblue'
    plot.plot('fill')

    plot.fontsize = 12
    #plot.color = 'gray'
    # dark gray
    plot.rgb = (0.3,0.3,0.3)
    if grid is not None:
        plot.plot_grid(*([grid]*4))

    plot.rgb = (0.4, 0.6, 0.4)
    ann = plot.annotations
    ann.NGC = ann.bright = ann.HD = 0
    ann.constellations = 1
    ann.constellation_labels = 1
    ann.constellation_labels_long = 1
    plot.plot('annotations')
    #plot.stroke()
    ann.constellation_labels = 0
    ann.constellation_lines = 0
    ann.constellation_markers = 1
    plot.markersize = 3
    plot.plot('annotations')
    plot.fill()

    if hd:
        ann.constellations=0
        ann.HD = 1
        ps.plot_annotations_set_hd_catalog(ann, settings.HENRY_DRAPER_CAT)
        plot.plot('annotations')
        plot.stroke()

    ann.constellations = 0
    ann.NGC = 1
    plot.plot('annotations')

    plot.color = 'white'
    plot.lw = 3
    out = plot.outline
    out.wcs_file = wcsfn
    plot.plot('outline')

    if zoom:
        # MAGIC width, height are arbitrary
        zoomwcs = anutil.anwcs_create_box(ra, dec, zoomwidth, 1000,1000)
        out.wcs = zoomwcs
        plot.lw = 1
        plot.dashed(3)
        plot.plot('outline')

    plot.write(plotfn)

def plot_aitoff_wcs_outline(wcsfn, plotfn, W=400, zoom=True):
    #anutil.log_init(3)
    H = W/2
    # Create Hammer-Aitoff WCS of the appropriate size.
    wcs = anutil.anwcs_create_allsky_hammer_aitoff(0., 0., W, H)

    plot = ps.Plotstuff(outformat='png', size=(W, H))
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

    # Not helpful to add constellations in this view
    #ann = plot.annotations
    #ann.NGC = ann.bright = ann.HD = 0
    #ann.constellations = 1
    #plot.plot('annotations')

    if zoom:
        owcs = anutil.Tan(wcsfn, 0)
        # MAGIC 15 degrees radius
        #if owcs.radius() < 15.:
        if True:
            ra,dec = owcs.radec_center()
            # MAGIC 36-degree width zoom-in
            # MAGIC width, height are arbitrary
            zoomwcs = anutil.anwcs_create_box(ra, dec, 36, 1000,1000)
            out.wcs = zoomwcs
            #plot.color = 'gray'
            plot.lw = 1
            plot.dashed(3)
            plot.plot('outline')

    plot.write(plotfn)
    
    
