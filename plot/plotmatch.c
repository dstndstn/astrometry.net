/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <string.h>
#include <math.h>

#include "plotmatch.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "permutedsort.h"
#include "matchfile.h"

DEFINE_PLOTTER(match);

plotmatch_t* plot_match_get(plot_args_t* pargs) {
    return plotstuff_get_config(pargs, "match");
}

void* plot_match_init(plot_args_t* plotargs) {
    plotmatch_t* args = calloc(1, sizeof(plotmatch_t));
    args->matches = bl_new(16, sizeof(MatchObj));
    return args;
}

int plot_match_plot(const char* command,
                    cairo_t* cairo, plot_args_t* pargs, void* baton) {
    plotmatch_t* args = (plotmatch_t*)baton;
    int i;
    anbool failed = FALSE;

    plotstuff_builtin_apply(cairo, pargs);

    for (i=0; i<bl_size(args->matches); i++) {
        MatchObj* mo = bl_access(args->matches, i);
        double xy[DQMAX*2];
        double theta[DQMAX];
        int perm[DQMAX];
        double cx,cy;
        int j;
        double x,y;
        //int order[] = {0,2,1,3,4,5};
        cx = cy = 0;
        for (j=0; j<mo->dimquads; j++) {
            double ra,dec;
            //xyzarr2radecdeg(mo->quadxyz + order[j]*3, &ra, &dec);
            xyzarr2radecdeg(mo->quadxyz + j*3, &ra, &dec);
            if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y)) {
                failed = TRUE;
                break;
            }
            xy[2*j + 0] = x;
            xy[2*j + 1] = y;
            cx += x;
            cy += y;
        }
        if (failed)
            continue;

        // (In the grand tradition of infinite copy-n-paste...)
        // Make the quad convex so Sam's eyes don't bleed.
        cx /= mo->dimquads;
        cy /= mo->dimquads;
        for (j=0; j<mo->dimquads; j++)
            theta[j] = atan2(xy[2*j + 1] - cy, xy[2*j + 0] - cx);
        permutation_init(perm, mo->dimquads);
        permuted_sort(theta, sizeof(double), compare_doubles_asc, perm, mo->dimquads);

        for (j=0; j<mo->dimquads; j++) {
            x = xy[2*perm[j]+0];
            y = xy[2*perm[j]+1];
            if (j == 0)
                cairo_move_to(cairo, x, y);
            else
                cairo_line_to(cairo, x, y);
        }
        cairo_close_path(cairo);
        cairo_stroke(cairo);
    }
    return 0;
}

int plot_match_add_match(plotmatch_t* args, const MatchObj* mo) {
    bl_append(args->matches, mo);
    return 0;
}

int plot_match_set_filename(plotmatch_t* args, const char* filename) {
    matchfile* mf = matchfile_open(filename);
    MatchObj* mo;
    if (!mf) {
        ERROR("Failed to open matchfile \"%s\"", filename);
        return -1;
    }
    while (1) {
        mo = matchfile_read_match(mf);
        if (!mo)
            break;
        plot_match_add_match(args, mo);
    }
    return 0;
}

int plot_match_command(const char* cmd, const char* cmdargs,
                       plot_args_t* pargs, void* baton) {
    plotmatch_t* args = (plotmatch_t*)baton;
    if (streq(cmd, "match_file")) {
        plot_match_set_filename(args, cmdargs);
    } else {
        ERROR("Did not understand command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

void plot_match_free(plot_args_t* plotargs, void* baton) {
    plotmatch_t* args = (plotmatch_t*)baton;
    bl_free(args->matches);
    free(args);
}

