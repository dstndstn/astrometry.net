/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <assert.h>

#include "os-features.h"
#include "plotxy.h"
#include "xylist.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"
#include "sip_qfits.h"
#include "tic.h"

DEFINE_PLOTTER(xy);

plotxy_t* plot_xy_get(plot_args_t* pargs) {
    return plotstuff_get_config(pargs, "xy");
}

void* plot_xy_init(plot_args_t* plotargs) {
    plotxy_t* args = calloc(1, sizeof(plotxy_t));
    args->ext = 1;
    args->scale = 1.0;
    args->xyvals = dl_new(32);
    // FITS pixels.
    args->xoff = 1.0;
    args->yoff = 1.0;
    return args;
}

int plot_xy_setsize(plot_args_t* pargs, plotxy_t* args) {
    xylist_t* xyls;
    xyls = xylist_open(args->fn);
    if (!xyls) {
        ERROR("Failed to open xylist from file \"%s\"", args->fn);
        return -1;
    }
    pargs->W = xylist_get_imagew(xyls);
    pargs->H = xylist_get_imageh(xyls);
    if (pargs->W == 0 && pargs->H == 0) {
        qfits_header* hdr = xylist_get_primary_header(xyls);
        pargs->W = qfits_header_getint(hdr, "IMAGEW", 0);
        pargs->H = qfits_header_getint(hdr, "IMAGEH", 0);
    }
    xylist_close(xyls);
    return 0;
}

int plot_xy_plot(const char* command, cairo_t* cairo,
                 plot_args_t* pargs, void* baton) {
    plotxy_t* args = (plotxy_t*)baton;
    // Plot it!
    xylist_t* xyls;
    starxy_t myxy;
    starxy_t* xy = NULL;
    starxy_t* freexy = NULL;
    int Nxy;
    int i;
#if 0
    double t0;
#endif

    plotstuff_builtin_apply(cairo, pargs);

    if (args->fn && dl_size(args->xyvals)) {
        ERROR("Can only plot one of xylist filename and xy_vals");
        return -1;
    }
    if (!args->fn && !dl_size(args->xyvals)) {
        ERROR("Neither xylist filename nor xy_vals given!");
        return -1;
    }

    if (args->fn) {
#if 0
        t0 = timenow();
#endif
        // Open xylist.
        xyls = xylist_open(args->fn);
        if (!xyls) {
            ERROR("Failed to open xylist from file \"%s\"", args->fn);
            return -1;
        }
        // we don't care about FLUX and BACKGROUND columns.
        xylist_set_include_flux(xyls, FALSE);
        xylist_set_include_background(xyls, FALSE);
        if (args->xcol)
            xylist_set_xname(xyls, args->xcol);
        if (args->ycol)
            xylist_set_yname(xyls, args->ycol);

        // Find number of entries in xylist.
        xy = xylist_read_field_num(xyls, args->ext, NULL);
        freexy = xy;
        xylist_close(xyls);
        if (!xy) {
            ERROR("Failed to read FITS extension %i from file %s.\n", args->ext, args->fn);
            return -1;
        }
        Nxy = starxy_n(xy);
        // If N is specified, apply it as a max.
        if (args->nobjs)
            Nxy = MIN(Nxy, args->nobjs);
        //logmsg("%g s to read xylist\n", timenow()-t0);
    } else {
        assert(dl_size(args->xyvals));
        starxy_from_dl(&myxy, args->xyvals, FALSE, FALSE);
        xy = &myxy;
        Nxy = starxy_n(xy);
    }

    // Transform through WCSes.
    if (args->wcs) {
        double ra, dec, x, y;
        assert(pargs->wcs);
        /*
         // check for any overlap.
         double pralo,prahi,pdeclo,pdechi;
         double ralo,rahi,declo,dechi;
         anwcs_get_radec_bounds(pargs->wcs, 100, &pralo, &prahi, &pdeclo, &pdechi);
         anwcs_get_radec_bounds(args->wcs, 100, &ralo, &rahi, &declo, &dechi);
         if (
         */
        for (i=0; i<Nxy; i++) {
            anwcs_pixelxy2radec(args->wcs,
                                // I used to add 1 here
                                starxy_getx(xy, i), starxy_gety(xy, i),
                                &ra, &dec);
            if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y))
                continue;
            logverb("  xy (%g,%g) -> RA,Dec (%g,%g) -> plot xy (%g,%g)\n",
                    starxy_getx(xy,i), starxy_gety(xy,i), ra, dec, x, y);

            // add shift and scale...
            // FIXME -- not clear that we want to do this here...
            /*
             starxy_setx(xy, i, args->scale * (x - args->xoff));
             starxy_sety(xy, i, args->scale * (y - args->yoff));
             starxy_setx(xy, i, x-1);
             starxy_sety(xy, i, y-1);
             */

            // Output coords: FITS -> 0-indexed image
            starxy_setx(xy, i, x-1);
            starxy_sety(xy, i, y-1);
        }
    } else {
        // Shift and scale xylist entries.
        if (args->xoff != 0.0 || args->yoff != 0.0) {
            for (i=0; i<Nxy; i++) {
                starxy_setx(xy, i, starxy_getx(xy, i) - args->xoff);
                starxy_sety(xy, i, starxy_gety(xy, i) - args->yoff);
            }
        }
        if (args->scale != 1.0) {
            for (i=0; i<Nxy; i++) {
                starxy_setx(xy, i, args->scale * starxy_getx(xy, i));
                starxy_sety(xy, i, args->scale * starxy_gety(xy, i));
            }
        }
    }

    // Plot markers.
#if 0
    t0 = timenow();
#endif
    for (i=args->firstobj; i<Nxy; i++) {
        double x = starxy_getx(xy, i);
        double y = starxy_gety(xy, i);
        if (plotstuff_marker_in_bounds(pargs, x, y))
            plotstuff_stack_marker(pargs, x, y);
    }
    plotstuff_plot_stack(pargs, cairo);
    //logmsg("%g s to plot xylist\n", timenow()-t0);

    starxy_free(freexy);
    return 0;
}

void plot_xy_set_xcol(plotxy_t* args, const char* col) {
    free(args->xcol);
    args->xcol = strdup_safe(col);
}

void plot_xy_set_ycol(plotxy_t* args, const char* col) {
    free(args->ycol);
    args->ycol = strdup_safe(col);
}

void plot_xy_set_filename(plotxy_t* args, const char* fn) {
    free(args->fn);
    args->fn = strdup_safe(fn);
}

int plot_xy_set_wcs_filename(plotxy_t* args, const char* fn, int ext) {
    anwcs_free(args->wcs);
    args->wcs = anwcs_open(fn, ext);
    if (!args->wcs) {
        ERROR("Failed to read WCS file \"%s\"", fn);
        return -1;
    }
    return 0;
}

int plot_xy_set_offsets(plotxy_t* args, double xo, double yo) {
    args->xoff = xo;
    args->yoff = yo;
    return 0;
}

void plot_xy_vals(plotxy_t* args, double x, double y) {
    dl_append(args->xyvals, x);
    dl_append(args->xyvals, y);
}

void plot_xy_clear_list(plotxy_t* args) {
    dl_remove_all(args->xyvals);
}

int plot_xy_command(const char* cmd, const char* cmdargs,
                    plot_args_t* plotargs, void* baton) {
    plotxy_t* args = (plotxy_t*)baton;
    if (streq(cmd, "xy_file")) {
        plot_xy_set_filename(args, cmdargs);
    } else if (streq(cmd, "xy_ext")) {
        args->ext = atoi(cmdargs);
    } else if (streq(cmd, "xy_xcol")) {
        plot_xy_set_xcol(args, cmdargs);
    } else if (streq(cmd, "xy_ycol")) {
        plot_xy_set_ycol(args, cmdargs);
    } else if (streq(cmd, "xy_xoff")) {
        args->xoff = atof(cmdargs);
    } else if (streq(cmd, "xy_yoff")) {
        args->yoff = atof(cmdargs);
    } else if (streq(cmd, "xy_firstobj")) {
        args->firstobj = atoi(cmdargs);
    } else if (streq(cmd, "xy_nobjs")) {
        args->nobjs = atoi(cmdargs);
    } else if (streq(cmd, "xy_scale")) {
        args->scale = atof(cmdargs);
        //} else if (streq(cmd, "xy_wcs")) {
        //return plot_xy_set_wcs_filename(args, cmdargs);
    } else if (streq(cmd, "xy_vals")) {
        plotstuff_append_doubles(cmdargs, args->xyvals);
    } else {
        ERROR("Did not understand command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

void plot_xy_free(plot_args_t* plotargs, void* baton) {
    plotxy_t* args = (plotxy_t*)baton;
    free(args->xyvals);
    anwcs_free(args->wcs);
    free(args->xcol);
    free(args->ycol);
    free(args->fn);
    free(args);
}

