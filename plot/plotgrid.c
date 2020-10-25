/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <string.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "plotgrid.h"
#include "sip-utils.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"

DEFINE_PLOTTER(grid);

plotgrid_t* plot_grid_get(plot_args_t* pargs) {
    return plotstuff_get_config(pargs, "grid");
}

void* plot_grid_init(plot_args_t* plotargs) {
    plotgrid_t* args = calloc(1, sizeof(plotgrid_t));
    args->dolabel = TRUE;
    args->raformat = strdup("%.2f");
    args->decformat = strdup("%+.2f");
    return args;
}

int plot_grid_set_formats(plotgrid_t* args, const char* raformat, const char* decformat) {
    free(args->raformat);
    free(args->decformat);
    args->raformat = strdup_safe(raformat);
    args->decformat = strdup_safe(decformat);
    return 0;
}

static void pretty_label(const char* fmt, double x, char* buf) {
    int i;
    sprintf(buf, fmt, x);
    logverb("label: \"%s\"\n", buf);
    // Look for decimal point.
    if (!strchr(buf, '.')) {
        logverb("no decimal point\n");
        return;
    }
    // Trim trailing zeroes (after the decimal point)
    i = strlen(buf)-1;
    while (buf[i] == '0') {
        buf[i] = '\0';
        logverb("trimming trailing zero at %i: \"%s\"\n", i, buf);
        i--;
        assert(i > 0);
    }
    // Trim trailing decimal point, if it exists.
    i = strlen(buf)-1;
    if (buf[i] == '.') {
        buf[i] = '\0';
        logverb("trimming trailing decimal point at %i: \"%s\"\n", i, buf);
    }
}

static int setdir(int dir, int* dirs, int* ndir) {
    switch (dir) {
    case DIRECTION_DEFAULT:
    case DIRECTION_POSNEG:
        dirs[0] = 1;
        dirs[1] = -1;
        *ndir = 2;
        break;
    case DIRECTION_POS:
        dirs[0] = 1;
        *ndir = 1;
        break;
    case DIRECTION_NEG:
        dirs[0] = -1;
        *ndir = 1;
        break;
    case DIRECTION_NEGPOS:
        dirs[0] = -1;
        dirs[1] = 1;
        *ndir = 2;
        break;
    default:
        return -1;
    }
    return 0;
}

int plot_grid_find_dec_label_location(plot_args_t* pargs, double dec, double cra, double ramin, double ramax, int dirn, double* pra) {
    double out;
    double in = cra;
    int i, N;
    anbool gotit;
    int dirs[2];
    int j, Ndir=0;
    logverb("Labelling Dec=%g\n", dec);
    gotit = FALSE;
    if (setdir(dirn, dirs, &Ndir))
        return -1;

    // dir is first 1, then -1.
    for (j=0; j<Ndir; j++) {
        int dir = dirs[j];
        for (i=1;; i++) {
            // take 10-deg steps
            out = cra + i*dir*10.0;
            if (out > 370.0 || out <= -10)
                break;
            out = MIN(360, MAX(0, out));
            logverb("ra in=%g, out=%g\n", in, out);
            if (!plotstuff_radec_is_inside_image(pargs, out, dec)) {
                gotit = TRUE;
                break;
            }
            if (!isfinite(in) || !isfinite(out))
                break;
        }
        if (gotit)
            break;
    }
    if (!gotit) {
        ERROR("Couldn't find an RA outside the image for Dec=%g\n", dec);
        return -1;
    }
    i=0;
    N = 10;
    while (!plotstuff_radec_is_inside_image(pargs, in, dec)) {
        if (i == N)
            break;
        in = ramin + (double)i/(double)(N-1) * (ramax-ramin);
        i++;
    }
    if (!plotstuff_radec_is_inside_image(pargs, in, dec))
        return -1;
    while (fabs(out - in) > 1e-6) {
        // hahaha
        double half;
        anbool isin;
        half = (out + in) / 2.0;
        isin = plotstuff_radec_is_inside_image(pargs, half, dec);
        if (isin)
            in = half;
        else
            out = half;
    }
    *pra = in;
    return 0;
}

int plot_grid_find_ra_label_location(plot_args_t* pargs, double ra, double cdec, double decmin, double decmax, int dirn, double* pdec) {
    double out;
    double in = cdec;
    int i, N;
    anbool gotit;
    int dirs[2];
    int j, Ndir=0;
    logverb("Labelling RA=%g\n", ra);
    // where does this line leave the image?
    // cdec is inside; take steps away until we go outside.
    gotit = FALSE;

    if (setdir(dirn, dirs, &Ndir))
        return -1;

    for (j=0; j<Ndir; j++) {
        int dir = dirs[j];
        logverb("direction: %i\n", dir);
        for (i=1;; i++) {
            // take 10-deg steps
            out = cdec + i*dir*10.0;
            logverb("trying Dec = %g\n", out);
            if (out >= 100.0 || out <= -100)
                break;
            out = MIN(90, MAX(-90, out));
            logverb("dec in=%g, out=%g\n", in, out);
            if (!plotstuff_radec_is_inside_image(pargs, ra, out)) {
                logverb("-> good!\n");
                gotit = TRUE;
                break;
            }
        }
        if (gotit)
            break;
    }
    if (!gotit) {
        ERROR("Couldn't find a Dec outside the image for RA=%g\n", ra);
        return -1;
    }
    // Now we've got a Dec inside the image (cdec)
    // and a Dec outside the image (out)
    // Now find the boundary.

    i=0;
    N = 10;
    while (!plotstuff_radec_is_inside_image(pargs, ra, in)) {
        if (i == N)
            break;
        in = decmin + (double)i/(double)(N-1) * (decmax-decmin);
        i++;
    }
    if (!plotstuff_radec_is_inside_image(pargs, ra, in))
        return -1;
    while (fabs(out - in) > 1e-6) {
        // hahaha
        double half;
        anbool isin;
        half = (out + in) / 2.0;
        isin = plotstuff_radec_is_inside_image(pargs, ra, half);
        if (isin)
            in = half;
        else
            out = half;
    }
    *pdec = in;
    return 0;
}

static int do_radec_labels(plot_args_t* pargs, plotgrid_t* args,
                           double ramin, double ramax,
                           double decmin, double decmax,
			   anbool doplot,
			   int* count_ra, int* count_dec) {
    double cra, cdec;
    double ra, dec;

    if (count_ra)
      *count_ra = 0;
    if (count_dec)
      *count_dec = 0;

    args->dolabel = (args->ralabelstep > 0) || (args->declabelstep > 0);
    if (!args->dolabel)
        return 0;

    /*
     if (args->ralabelstep == 0 || args->declabelstep == 0) {
     // FIXME -- choose defaults
     ERROR("Need grid_ralabelstep, grid_declabelstep");
     return 0;
     }
     */
    logmsg("Adding grid labels...\n");
    plotstuff_get_radec_center_and_radius(pargs, &cra, &cdec, NULL);
    assert(cra >= ramin && cra <= ramax);
    assert(cdec >= decmin && cdec <= decmax);
    if (args->ralabelstep > 0) {
        double rlo, rhi;
        if (args->ralo != 0 || args->rahi != 0) {
            rlo = args->ralo;
            rhi = args->rahi;
        } else {
            rlo = args->ralabelstep * floor(ramin / args->ralabelstep);
            rhi = args->ralabelstep * ceil(ramax / args->ralabelstep);
        }
        for (ra = rlo; ra <= rhi; ra += args->ralabelstep) {
            double lra;
            if (plot_grid_find_ra_label_location(pargs, ra, cdec, decmin,
                                                 decmax, args->ralabeldir, &dec))
                continue;
            lra = ra;
            if (lra < 0)
                lra += 360;
            if (lra >= 360)
                lra -= 360;

	    if (count_ra)
	      (*count_ra)++;
	    if (doplot)
	      plot_grid_add_label(pargs, ra, dec, lra, args->raformat);
        }
    }
    if (args->declabelstep > 0) {
        double dlo, dhi;
        if (args->declo != 0 || args->dechi != 0) {
            dlo = args->declo;
            dhi = args->dechi;
        } else {
            dlo = args->declabelstep * floor(decmin / args->declabelstep);
            dhi = args->declabelstep * ceil(decmax / args->declabelstep);
        }
        for (dec = dlo; dec <= dhi; dec += args->declabelstep) {
            if (plot_grid_find_dec_label_location(pargs, dec, cra, ramin,
                                                  ramax, args->declabeldir, &ra))
                continue;
	    if (count_dec)
	      (*count_dec)++;
	    if (doplot)
	      plot_grid_add_label(pargs, ra, dec, dec, args->decformat);
        }
    }
    return 1;
}

// With the current "ralabelstep", how many labels will be added?
int plot_grid_count_ra_labels(plot_args_t* pargs) {
  plotgrid_t* grid = plot_grid_get(pargs);
  int count;
  double ramin,ramax,decmin,decmax;
  if (!pargs->wcs)
    return -1;
  // Find image bounds in RA,Dec...
  plotstuff_get_radec_bounds(pargs, 50, &ramin, &ramax, &decmin, &decmax);
  do_radec_labels(pargs, grid, ramin, ramax, decmin, decmax, FALSE, &count, NULL);
  return count;
}
// With the current "declabelstep", how many labels will be added?
int plot_grid_count_dec_labels(plot_args_t* pargs) {
  plotgrid_t* grid = plot_grid_get(pargs);
  int count;
  double ramin,ramax,decmin,decmax;
  if (!pargs->wcs)
    return -1;
  // Find image bounds in RA,Dec...
  plotstuff_get_radec_bounds(pargs, 50, &ramin, &ramax, &decmin, &decmax);
 do_radec_labels(pargs, grid, ramin, ramax, decmin, decmax, FALSE, NULL, &count);
  return count;
}

void plot_grid_add_label(plot_args_t* pargs, double ra, double dec,
                         double lval, const char* format) {
    char label[32];
    double x,y;
#if 0
    anbool ok;
#endif
    cairo_t* cairo = pargs->cairo;
    pretty_label(format, lval, label);
#if 0
    ok = plotstuff_radec2xy(pargs, ra, dec, &x, &y);
#else
    (void)plotstuff_radec2xy(pargs, ra, dec, &x, &y);
#endif
    plotstuff_stack_text(pargs, cairo, label, x, y);
    plotstuff_plot_stack(pargs, cairo);
}

int plot_grid_plot(const char* command,
                   cairo_t* cairo, plot_args_t* pargs, void* baton) {
    plotgrid_t* args = (plotgrid_t*)baton;
    double ramin,ramax,decmin,decmax;
    double ra,dec;

    if (!pargs->wcs) {
        ERROR("No WCS was set -- can't plot grid lines");
        return -1;
    }
    // Find image bounds in RA,Dec...
    plotstuff_get_radec_bounds(pargs, 50, &ramin, &ramax, &decmin, &decmax);
    /*
     if (args->rastep == 0 || args->decstep == 0) {
     // FIXME -- choose defaults
     ERROR("Need grid_rastep, grid_decstep");
     return -1;
     }
     */

    plotstuff_builtin_apply(cairo, pargs);
    pargs->label_offset_x = 0;
    pargs->label_offset_y = 10;
	
    logverb("Image bounds: RA %g, %g, Dec %g, %g\n",
            ramin, ramax, decmin, decmax);
    if (args->rastep > 0) {
        for (ra = args->rastep * floor(ramin / args->rastep);
             ra <= args->rastep * ceil(ramax / args->rastep);
             ra += args->rastep) {
            plotstuff_line_constant_ra(pargs, ra, decmin, decmax, TRUE);
            cairo_stroke(pargs->cairo);
        }
    }
    if (args->decstep > 0) {
        for (dec = args->decstep * floor(decmin / args->decstep);
             dec <= args->decstep * ceil(decmax / args->decstep);
             dec += args->decstep) {
            plotstuff_line_constant_dec(pargs, dec, ramin, ramax);
            cairo_stroke(pargs->cairo);
        }
    }

    if (do_radec_labels(pargs, args, ramin, ramax, decmin, decmax,
			TRUE, NULL, NULL)) {
        plotstuff_plot_stack(pargs, cairo);
    }
    return 0;
}




int plot_grid_command(const char* cmd, const char* cmdargs,
                      plot_args_t* pargs, void* baton) {
    plotgrid_t* args = (plotgrid_t*)baton;
    if (streq(cmd, "grid_rastep")) {
        args->rastep = atof(cmdargs);
    } else if (streq(cmd, "grid_decstep")) {
        args->decstep = atof(cmdargs);
    } else if (streq(cmd, "grid_ralabelstep")) {
        args->ralabelstep = atof(cmdargs);
    } else if (streq(cmd, "grid_declabelstep")) {
        args->declabelstep = atof(cmdargs);
    } else if (streq(cmd, "grid_step")) {
        args->declabelstep = args->ralabelstep =
            args->rastep = args->decstep = atof(cmdargs);
    } else {
        ERROR("Did not understand command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

void plot_grid_free(plot_args_t* plotargs, void* baton) {
    plotgrid_t* args = (plotgrid_t*)baton;
    free(args->raformat);
    free(args->decformat);
    free(args);
}

