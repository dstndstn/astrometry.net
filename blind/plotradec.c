/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <assert.h>

#include "os-features.h"
#include "plotradec.h"
#include "rdlist.h"
#include "cairoutils.h"
#include "log.h"
#include "errors.h"
#include "sip_qfits.h"

DEFINE_PLOTTER(radec);

plotradec_t* plot_radec_get(plot_args_t* pargs) {
    return plotstuff_get_config(pargs, "radec");
}

void plot_radec_reset(plotradec_t* args) {
    if (args->radecvals)
        dl_free(args->radecvals);
    if (args->racol)
        free(args->racol);
    if (args->deccol)
        free(args->deccol);
    if (args->fn)
        free(args->fn);
    memset(args, 0, sizeof(plotradec_t));
    args->ext = 1;
    args->radecvals = dl_new(32);
}

void* plot_radec_init(plot_args_t* plotargs) {
    plotradec_t* args = calloc(1, sizeof(plotradec_t));
    plot_radec_reset(args);
    return args;
}

static rd_t* get_rd(plotradec_t* args, rd_t* myrd) {
    rdlist_t* rdls = NULL;
    rd_t* rd = NULL;
    if (args->fn) {
        // Open rdlist.
        rdls = rdlist_open(args->fn);
        if (!rdls) {
            ERROR("Failed to open rdlist from file \"%s\"", args->fn);
            return NULL;
        }
        if (args->racol)
            rdlist_set_raname(rdls, args->racol);
        if (args->deccol)
            rdlist_set_decname(rdls, args->deccol);

        // Find number of entries in rdlist.
        rd = rdlist_read_field_num(rdls, args->ext, NULL);
        //freerd = rd;
        rdlist_close(rdls);
        if (!rd) {
            ERROR("Failed to read FITS extension %i from file %s.\n", args->ext, args->fn);
            return NULL;
        }
    } else {
        assert(dl_size(args->radecvals));
        rd_from_dl(myrd, args->radecvals);
        rd = myrd;
    }
    return rd;
}

int plot_radec_plot(const char* command, cairo_t* cairo,
                    plot_args_t* pargs, void* baton) {
    plotradec_t* args = (plotradec_t*)baton;
    // Plot it!
    rd_t myrd;
    rd_t* rd = NULL;
    //rd_t* freerd = NULL;
    int Nrd;
    int i;

    if (!pargs->wcs) {
        ERROR("plotting radec but not plot_wcs has been set.");
        return -1;
    }

    if (args->fn && dl_size(args->radecvals)) {
        ERROR("Can only plot one of rdlist filename and radec_vals");
        return -1;
    }
    if (!args->fn && !dl_size(args->radecvals)) {
        ERROR("Neither rdlist filename nor radec_vals given!");
        return -1;
    }

    plotstuff_builtin_apply(cairo, pargs);

    rd = get_rd(args, &myrd);
    if (!rd) return -1;
    Nrd = rd_n(rd);
    // If N is specified, apply it as a max.
    if (args->nobjs)
        Nrd = MIN(Nrd, args->nobjs);

    // Plot markers.
    for (i=args->firstobj; i<Nrd; i++) {
        double x,y;
        double ra = rd_getra(rd, i);
        double dec = rd_getdec(rd, i);
        if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y))
            continue;
        if (!plotstuff_marker_in_bounds(pargs, x, y))
            continue;
        plotstuff_stack_marker(pargs, x-1, y-1);
    }
    plotstuff_plot_stack(pargs, cairo);

    if (rd != &myrd)
        rd_free(rd);
    //rd_free(freerd);
    return 0;
}

int plot_radec_count_inbounds(plot_args_t* pargs, plotradec_t* args) {
    rd_t myrd;
    rd_t* rd = NULL;
    int i, Nrd, nib;

    rd = get_rd(args, &myrd);
    if (!rd) return -1;
    Nrd = rd_n(rd);
    // If N is specified, apply it as a max.
    if (args->nobjs)
        Nrd = MIN(Nrd, args->nobjs);
    nib = 0;
    for (i=args->firstobj; i<Nrd; i++) {
        double x,y;
        double ra = rd_getra(rd, i);
        double dec = rd_getdec(rd, i);
        if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y))
            continue;
        if (!plotstuff_marker_in_bounds(pargs, x, y))
            continue;
        nib++;
    }
    if (rd != &myrd)
        rd_free(rd);
    return nib;
}

void plot_radec_set_racol(plotradec_t* args, const char* col) {
    free(args->racol);
    args->racol = strdup_safe(col);
}

void plot_radec_set_deccol(plotradec_t* args, const char* col) {
    free(args->deccol);
    args->deccol = strdup_safe(col);
}

void plot_radec_set_filename(plotradec_t* args, const char* fn) {
    free(args->fn);
    args->fn = strdup_safe(fn);
}

void plot_radec_vals(plotradec_t* args, double ra, double dec) {
    dl_append(args->radecvals, ra);
    dl_append(args->radecvals, dec);
}

int plot_radec_command(const char* cmd, const char* cmdargs,
                       plot_args_t* plotargs, void* baton) {
    plotradec_t* args = (plotradec_t*)baton;
    if (streq(cmd, "radec_file")) {
        plot_radec_set_filename(args, cmdargs);
    } else if (streq(cmd, "radec_ext")) {
        args->ext = atoi(cmdargs);
    } else if (streq(cmd, "radec_racol")) {
        plot_radec_set_racol(args, cmdargs);
    } else if (streq(cmd, "radec_deccol")) {
        plot_radec_set_deccol(args, cmdargs);
    } else if (streq(cmd, "radec_firstobj")) {
        args->firstobj = atoi(cmdargs);
    } else if (streq(cmd, "radec_nobjs")) {
        args->nobjs = atoi(cmdargs);
    } else if (streq(cmd, "radec_vals")) {
        plotstuff_append_doubles(cmdargs, args->radecvals);
    } else {
        ERROR("Did not understand command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

void plot_radec_free(plot_args_t* plotargs, void* baton) {
    plotradec_t* args = (plotradec_t*)baton;
    free(args->radecvals);
    free(args->racol);
    free(args->deccol);
    free(args->fn);
    free(args);
}

