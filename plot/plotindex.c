/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <string.h>
#include <math.h>
#include <assert.h>

#include "plotindex.h"
#include "cairoutils.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "starutil.h"
#include "index.h"
#include "qidxfile.h"
#include "permutedsort.h"

DEFINE_PLOTTER(index);

plotindex_t* plot_index_get(plot_args_t* pargs) {
    return plotstuff_get_config(pargs, "index");
}

void* plot_index_init(plot_args_t* plotargs) {
    plotindex_t* args = calloc(1, sizeof(plotindex_t));
    args->indexes = pl_new(16);
    args->qidxes = pl_new(16);
    args->stars = TRUE;
    args->quads = TRUE;
    args->fill = FALSE;
    return args;
}

static void pad_qidxes(plotindex_t* args) {
    while ((pl_size(args->qidxes)) < pl_size(args->indexes))
        pl_append(args->qidxes, NULL);
}

void plot_quad_xy(cairo_t* cairo, double* quadxy, int dimquads) {
    int k;
    double cx, cy;
    double theta[DQMAX];
    int* perm;

    cx = cy = 0.0;
    for (k=0; k<dimquads; k++) {
        cx += quadxy[2*k+0];
        cy += quadxy[2*k+1];
    }
    cx /= dimquads;
    cy /= dimquads;

    // initialize to avoid compiler warning
    for (k=0; k<DQMAX; k++)
        theta[k] = 0.;
    for (k=0; k<dimquads; k++)
        theta[k] = atan2(quadxy[2*k+1] - cy, quadxy[2*k+0] - cx);
    perm = permuted_sort(theta, sizeof(double), compare_doubles_asc, NULL, dimquads);
    for (k=0; k<dimquads; k++) {
        double px,py;
        px = quadxy[2 * perm[k] + 0];
        py = quadxy[2 * perm[k] + 1];
        if (k == 0) {
            cairo_move_to(cairo, px, py);
        } else {
            cairo_line_to(cairo, px, py);
        }
    }
    free(perm);
    cairo_close_path(cairo);
}

static void plotquad(cairo_t* cairo, plot_args_t* pargs, plotindex_t* args, index_t* index, int quadnum, int DQ) {
    int k;
    unsigned int stars[DQMAX];
    double ra, dec;
    double px, py;
    double xy[DQMAX*2];
    int N;

    quadfile_get_stars(index->quads, quadnum, stars);
    N = 0;
    for (k=0; k<DQ; k++) {
        if (startree_get_radec(index->starkd, stars[k], &ra, &dec)) {
            ERROR("Failed to get RA,Dec for star %i\n", stars[k]);
            continue;
        }
        if (!plotstuff_radec2xy(pargs, ra, dec, &px, &py)) {
            ERROR("Failed to convert RA,Dec %g,%g to pixels for quad %i\n", ra, dec, quadnum);
            continue;
        }
        xy[2*k + 0] = px;
        xy[2*k + 1] = py;
        N++;
    }
    if (N < 3)
        return;
    plot_quad_xy(cairo, xy, N);
    if (args->fill)
        cairo_fill(cairo);
    else
        cairo_stroke(cairo);
}

void plot_index_plotquad(cairo_t* cairo, plot_args_t* pargs, plotindex_t* args, index_t* index, int quadnum, int DQ) {
    plotquad(cairo, pargs, args, index, quadnum, DQ);
}

int plot_index_plot(const char* command,
                    cairo_t* cairo, plot_args_t* pargs, void* baton) {
    plotindex_t* args = (plotindex_t*)baton;
    int i;
    double ra, dec, radius;
    double xyz[3];
    double r2;

    pad_qidxes(args);

    plotstuff_builtin_apply(cairo, pargs);

    if (plotstuff_get_radec_center_and_radius(pargs, &ra, &dec, &radius)) {
        ERROR("Failed to get RA,Dec center and radius");
        return -1;
    }
    radecdeg2xyzarr(ra, dec, xyz);
    r2 = deg2distsq(radius);
    logmsg("Field RA,Dec,radius = (%g,%g), %g deg\n", ra, dec, radius);
    logmsg("distsq: %g\n", r2);

    for (i=0; i<pl_size(args->indexes); i++) {
        index_t* index = pl_get(args->indexes, i);
        int j, N;
        int DQ;
        double px,py;

        if (args->stars) {
            // plot stars
            double* radecs = NULL;
            startree_search_for(index->starkd, xyz, r2, NULL, &radecs, NULL, &N);
            if (N) {
                assert(radecs);
            }
            logmsg("Found %i stars in range in index %s\n", N, index->indexname);
            for (j=0; j<N; j++) {
                logverb("  RA,Dec (%g,%g) -> x,y (%g,%g)\n", radecs[2*j], radecs[2*j+1], px, py);
                if (!plotstuff_radec2xy(pargs, radecs[j*2], radecs[j*2+1], &px, &py)) {
                    ERROR("Failed to convert RA,Dec %g,%g to pixels\n", radecs[j*2], radecs[j*2+1]);
                    continue;
                }
                cairoutils_draw_marker(cairo, pargs->marker, px, py, pargs->markersize);
                cairo_stroke(cairo);
            }
            free(radecs);
        }
        if (args->quads) {
            DQ = index_get_quad_dim(index);
            qidxfile* qidx = pl_get(args->qidxes, i);
            if (qidx) {
                int* stars;
                int Nstars;
                il* quadlist = il_new(256);

                // find stars in range.
                startree_search_for(index->starkd, xyz, r2, NULL, NULL, &stars, &Nstars);
                logmsg("Found %i stars in range of index %s\n", N, index->indexname);
                logmsg("Using qidx file.\n");
                // find quads that each star is a member of.
                for (j=0; j<Nstars; j++) {
                    uint32_t* quads;
                    int Nquads;
                    int k;
                    if (qidxfile_get_quads(qidx, stars[j], &quads, &Nquads)) {
                        ERROR("Failed to get quads for star %i\n", stars[j]);
                        return -1;
                    }
                    for (k=0; k<Nquads; k++)
                        il_insert_unique_ascending(quadlist, quads[k]);
                }
                for (j=0; j<il_size(quadlist); j++) {
                    plotquad(cairo, pargs, args, index, il_get(quadlist, j), DQ);
                }

            } else {
                // plot quads
                N = index_nquads(index);
                for (j=0; j<N; j++) {
                    plotquad(cairo, pargs, args, index, j, DQ);
                }
            }
        }
    }
    return 0;
}

int plot_index_add_qidx_file(plotindex_t* args, const char* fn) {
    int i;
    qidxfile* qidx = qidxfile_open(fn);
    if (!qidx) {
        ERROR("Failed to open quad index file \"%s\"", fn);
        return -1;
    }
    pad_qidxes(args);
    i = pl_size(args->indexes) - 1;
    pl_set(args->qidxes, i, qidx);
    return 0;
}

int plot_index_add_file(plotindex_t* args, const char* fn) {
    index_t* index = index_load(fn, 0, NULL);
    if (!index) {
        ERROR("Failed to open index \"%s\"", fn);
        return -1;
    }
    pl_append(args->indexes, index);
    return 0;
}

int plot_index_command(const char* cmd, const char* cmdargs,
                       plot_args_t* pargs, void* baton) {
    plotindex_t* args = (plotindex_t*)baton;
    if (streq(cmd, "index_file")) {
        const char* fn = cmdargs;
        return plot_index_add_file(args, fn);
    } else if (streq(cmd, "index_qidxfile")) {
        const char* fn = cmdargs;
        return plot_index_add_qidx_file(args, fn);
    } else if (streq(cmd, "index_draw_stars")) {
        args->stars = atoi(cmdargs);
    } else if (streq(cmd, "index_draw_quads")) {
        args->quads = atoi(cmdargs);
    } else if (streq(cmd, "index_fill")) {
        args->fill = atoi(cmdargs);
    } else {
        ERROR("Did not understand command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

void plot_index_free(plot_args_t* plotargs, void* baton) {
    plotindex_t* args = (plotindex_t*)baton;
    int i;
    for (i=0; i<pl_size(args->indexes); i++) {
        index_t* index = pl_get(args->indexes, i);
        index_free(index);
    }
    pl_free(args->indexes);
    for (i=0; i<pl_size(args->qidxes); i++) {
        qidxfile* qidx = pl_get(args->qidxes, i);
        qidxfile_close(qidx);
    }
    pl_free(args->qidxes);
    free(args);
}

