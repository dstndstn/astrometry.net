/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "kdtree.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "bl-sort.h"
#include "matchobj.h"
#include "catalog.h"
#include "tic.h"
#include "quadfile.h"
#include "quad-utils.h"
#include "xylist.h"
#include "rdlist.h"
#include "qidxfile.h"
#include "verify.h"
#include "ioutils.h"
#include "starkd.h"
#include "codekd.h"
#include "index.h"
#include "boilerplate.h"
#include "sip.h"
#include "sip-utils.h"
#include "sip_qfits.h"
#include "log.h"
#include "fitsioutils.h"
#include "fit-wcs.h"
#include "codefile.h"
#include "solver.h"
#include "permutedsort.h"
#include "intmap.h"

#include "plotstuff.h"
#include "plotxy.h"
#include "plotindex.h"

static const char* OPTIONS = "hx:w:i:vj:X:Y:Q:s:O:";

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "   -w <WCS input file>\n"
           "   -x <xyls input file>\n"
           "   -i <index-name>\n"
           "   [-X <x-column>]: X column name\n"
           "   [-Y <y-column>]: Y column name\n"
           "   [-v]: verbose\n"
           "   [-j <pixel-jitter>]: set pixel jitter (default 1.0)\n"
           "   [-Q <quad-plot-basename>]: plot all the quads in the image, one per file.  Put a %%i in this filename!\n"
           "   [-O <obj-plot-basename>]: make a plot for each image object; Put a %%i in this filename\n"
           "   [-s <plot-scale>]: scale everything by this factor (eg, 0.25)\n"
           "\n", progname);
}


struct foundquad {
    //unsigned int stars[DQMAX];
    double codedist;
    double logodds;
    double pscale;
    int quadnum;
    MatchObj mo;
};
typedef struct foundquad foundquad_t;

static int sort_fq_by_stars(const void* v1, const void* v2) {
    const foundquad_t* fq1 = v1;
    const foundquad_t* fq2 = v2;
    int mx1=0, mx2=0;
    int i;
    for (i=0; i<DQMAX; i++) {
        mx1 = MAX(mx1, fq1->mo.field[i]);
        mx2 = MAX(mx2, fq2->mo.field[i]);
    }
    if (mx1 < mx2)
        return -1;
    if (mx1 == mx2)
        return 0;
    return 1;
}

struct correspondence {
    int star;
    int field;
    double starx, stary;
    double fieldx, fieldy;
    double dist;
};
typedef struct correspondence corr_t;

struct indexstar {
    int star;
    double starx, stary;
    pl* corrs;
    il* allquads;
};
typedef struct indexstar indexstar_t;


int main(int argc, char** args) {
    int c;
    char* xylsfn = NULL;
    char* wcsfn = NULL;

    sl* indexnames;
    pl* indexes;
    pl* qidxes;

    xylist_t* xyls = NULL;
    sip_t sip;
    int i;
    int W, H;
    double xyzcenter[3];
    double fieldrad2;
    double pixeljitter = 1.0;
    int loglvl = LOG_MSG;
    double wcsscale;

    char* xcol = NULL;
    char* ycol = NULL;
	
    double nsigma = 3.0;

    char* quadplotfn = NULL;
    char* objplotfn = NULL;
    double quadplotscale = 1.0;

    fits_use_error_system();

    indexnames = sl_new(8);

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 's':
            quadplotscale = atof(optarg);
            break;
        case 'Q':
            quadplotfn = optarg;
            break;
        case 'O':
            objplotfn = optarg;
            break;
        case 'X':
            xcol = optarg;
            break;
        case 'Y':
            ycol = optarg;
            break;
        case 'j':
            pixeljitter = atof(optarg);
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'i':
            sl_append(indexnames, optarg);
            break;
        case 'x':
            xylsfn = optarg;
            break;
        case 'w':
            wcsfn = optarg;
            break;
        case 'v':
            loglvl++;
            break;
        }
    }
    if (optind != argc) {
        print_help(args[0]);
        exit(-1);
    }
    if (!xylsfn || !wcsfn) {
        print_help(args[0]);
        exit(-1);
    }
    log_init(loglvl);

    // read WCS.
    logmsg("Trying to parse SIP header from %s...\n", wcsfn);
    if (!sip_read_header_file(wcsfn, &sip)) {
        logmsg("Failed to parse SIP header from %s.\n", wcsfn);
    }
    // image W, H
    W = sip.wcstan.imagew;
    H = sip.wcstan.imageh;
    if ((W == 0.0) || (H == 0.0)) {
        logmsg("WCS file %s didn't contain IMAGEW and IMAGEH headers.\n", wcsfn);
        // FIXME - use bounds of xylist?
        exit(-1);
    }
    wcsscale = sip_pixel_scale(&sip);
    logmsg("WCS scale: %g arcsec/pixel\n", wcsscale);

    // read XYLS.
    xyls = xylist_open(xylsfn);
    if (!xyls) {
        logmsg("Failed to read an xylist from file %s.\n", xylsfn);
        exit(-1);
    }
    if (xcol)
        xylist_set_xname(xyls, xcol);
    if (ycol)
        xylist_set_yname(xyls, ycol);
    xylist_set_include_flux(xyls, FALSE);
    xylist_set_include_background(xyls, FALSE);



    // read indices.
    indexes = pl_new(8);
    qidxes = pl_new(8);
    for (i=0; i<sl_size(indexnames); i++) {
        char* name = sl_get(indexnames, i);
        index_t* indx;
        char* qidxfn;
        qidxfile* qidx;
        logmsg("Loading index from %s...\n", name);
        indx = index_load(name, 0, NULL);
        if (!indx) {
            logmsg("Failed to read index \"%s\".\n", name);
            exit(-1);
        }
        pl_append(indexes, indx);

        logmsg("Index name: %s\n", indx->indexname);

        qidxfn = index_get_qidx_filename(indx->indexname);
        qidx = qidxfile_open(qidxfn);
        if (!qidx) {
            logmsg("Failed to open qidxfile \"%s\".\n", qidxfn);
            exit(-1);            
        }
        free(qidxfn);
        pl_append(qidxes, qidx);
    }
    sl_free2(indexnames);

    // Find field center and radius.
    sip_pixelxy2xyzarr(&sip, W/2, H/2, xyzcenter);
    fieldrad2 = arcsec2distsq(sip_pixel_scale(&sip) * hypot(W/2, H/2));

    // Find all stars in the field.
    for (i=0; i<pl_size(indexes); i++) {
        index_t* indx;
        int nquads;
        uint32_t* quads;
        int j, k;
        qidxfile* qidx;
        il* uniqquadlist;
        bl* foundquads = NULL;

        // index stars that are inside the image.
        il* starlist;

        // index stars that have correspondences.
        il* corrstarlist;

        // quads that are at least partly-contained in the image.
        il* quadlist;

        // quads that are fully-contained in the image.
        il* fullquadlist;

        // index stars that are in partly-contained quads.
        il* starsinquadslist;

        // index stars that are in fully-contained quads.
        il* starsinquadsfull;

        // index stars that are in quads and have correspondences.
        il* corrstars;
        // the corresponding field stars
        il* corrfield;

        // quads that are fully in the image and built from stars with correspondences.
        il* corrfullquads;


        // Map from index stars to list of field stars corresponding.
        //intmap_t* corr_i2f;
        //intmap_t* corr_f2i;

        bl* corrs;

        indexstar_t* istars;
	
        dl* starxylist;
        il* corrquads;
        il* corruniqquads;
        starxy_t* xy;

        // (x,y) positions of field stars.
        double* fieldxy;

        int Nfield;
        kdtree_t* ftree;
        kdtree_t* itree;
        int Nleaf = 5;
        int dimquads, dimcodes;
        int ncorr, nindexcorr;
        double pixr2;
        il* indstarswithcorrs;

        double* xyzs = NULL;

        // Indices (in the star-kdtree) of stars that are within the image rectangle;
        // Ngood of them.
        int* starinds = NULL;

        int Nindex;
        int Ngood;

        indx = pl_get(indexes, i);
        qidx = pl_get(qidxes, i);

        corrs = bl_new(256, sizeof(corr_t));

        logmsg("Index jitter: %g arcsec (%g pixels)\n", indx->index_jitter, indx->index_jitter / wcsscale);
        pixr2 = square(indx->index_jitter / wcsscale) + square(pixeljitter);
        logmsg("Total jitter: %g pixels\n", sqrt(pixr2));

        // Read field
        xy = xylist_read_field(xyls, NULL);
        if (!xy) {
            logmsg("Failed to read xyls entries.\n");
            exit(-1);
        }
        Nfield = starxy_n(xy);
        fieldxy = starxy_to_xy_array(xy, NULL);
        logmsg("Found %i field objects\n", Nfield);

        // Find index stars.
        startree_search_for(indx->starkd, xyzcenter, fieldrad2*1.05,
                            &xyzs, NULL, &starinds, &Nindex);
        if (!Nindex) {
            logmsg("No index stars found.\n");
            exit(-1);
        }
        logmsg("Found %i index stars in range.\n", Nindex);

        starlist = il_new(16);
        corrstarlist = il_new(16);
        starxylist = dl_new(16);

        // Find which ones in range are inside the image rectangle.
        Ngood = 0;
        for (j=0; j<Nindex; j++) {
            int starnum = starinds[j];
            double x, y;
            if (!sip_xyzarr2pixelxy(&sip, xyzs + j*3, &x, &y))
                continue;
            if ((x < 0) || (y < 0) || (x >= W) || (y >= H))
                continue;
            il_append(starlist, starnum);
            dl_append(starxylist, x);
            dl_append(starxylist, y);
            // compact this array...
            starinds[Ngood] = starnum;
            Ngood++;
        }
        logmsg("Found %i index stars inside the field.\n", (int)il_size(starlist));

        assert(Ngood == il_size(starlist));
        assert(Ngood * 2 == dl_size(starxylist));

        istars = malloc(Ngood * sizeof(indexstar_t));

        // Now find correspondences between index objects and field objects.
        // Build a tree out of the field objects (in pixel space)
        // We make a copy of fieldxy because the copy gets permuted in this process.
        {
            double* fxycopy = malloc(Nfield * 2 * sizeof(double));
            memcpy(fxycopy, fieldxy, Nfield * 2 * sizeof(double));
            ftree = kdtree_build(NULL, fxycopy, Nfield, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);
        }
        if (!ftree) {
            logmsg("Failed to build kdtree.\n");
            exit(-1);
        }

        // Search for correspondences
        /*
         corr_i2f = intmap_new(sizeof(int), 4, 256, 0);
         corr_f2i = intmap_new(sizeof(int), 4, 256, 0);
         */
        ncorr = 0;
        nindexcorr = 0;
        indstarswithcorrs = il_new(16);
        for (j=0; j<dl_size(starxylist)/2; j++) {
            double xy[2];
            kdtree_qres_t* res;
            int nn;
            double nnd2;
            corr_t c;
            xy[0] = dl_get(starxylist, j*2+0);
            xy[1] = dl_get(starxylist, j*2+1);

            // kdtree check.
            /*
             int k;
             for (k=0; k<Nfield; k++) {
             double d2 = distsq(fieldxy + 2*k, xy, 2);
             if (d2 < pixr2) {
             logverb("  index star at (%.1f, %.1f) and field star at (%.1f, %.1f)\n", xy[0], xy[1],
             fieldxy[2*k+0], fieldxy[2*k+1]);
             }
             }
             */

            nn = kdtree_nearest_neighbour(ftree, xy, &nnd2);
            logverb("  Index star at (%.1f, %.1f): nearest field star is %g pixels away.\n",
                    xy[0], xy[1], sqrt(nnd2));

            res = kdtree_rangesearch_options(ftree, xy, pixr2 * nsigma*nsigma, KD_OPTIONS_SMALL_RADIUS | KD_OPTIONS_COMPUTE_DISTS | KD_OPTIONS_SORT_DISTS);

            istars[j].star = il_get(starlist, j);
            istars[j].starx = dl_get(starxylist, 2*j+0);
            istars[j].stary = dl_get(starxylist, 2*j+1);
            istars[j].corrs = pl_new(16);
            istars[j].allquads = il_new(16);

            if (!res || !res->nres)
                continue;

            for (k=0; k<res->nres; k++) {
                corr_t* cc;
                memset(&c, 0, sizeof(c));
                c.star = il_get(starlist, j);
                c.field = res->inds[k];
                c.starx = dl_get(starxylist, 2*j+0);
                c.stary = dl_get(starxylist, 2*j+1);
                c.fieldx = fieldxy[2*res->inds[k]+0];
                c.fieldy = fieldxy[2*res->inds[k]+1];
                c.dist = sqrt(res->sdists[k]);
                // add to master list, recording ptr.
                cc = bl_append(corrs, &c);
                pl_append(istars[j].corrs, cc);
            }

            /*
             for (k=0; k<res->nres; k++) {
             intmap_append(corr_i2f, il_get(starlist, j), res->inds + k);
             }
             */

            ncorr += res->nres;
            nindexcorr++;
            kdtree_free_query(res);
            il_append(indstarswithcorrs, j);

            il_append(corrstarlist, il_get(starlist, j));
        }
        logmsg("Found %i index stars with corresponding field stars.\n", nindexcorr);
        logmsg("Found %i total index star correspondences\n", ncorr);

        {
            double* ixycopy = malloc(Ngood * 2 * sizeof(double));
            dl_copy(starxylist, 0, Ngood * 2, ixycopy);
            itree = kdtree_build(NULL, ixycopy, Ngood, 2, Nleaf, KDTT_DOUBLE, KD_BUILD_SPLIT);
        }
        if (!itree) {
            logmsg("Failed to build index kdtree.\n");
            exit(-1);
        }
        for (j=0; j<Nfield; j++) {
            double* fxy;
            int nn;
            kdtree_qres_t* res;
            double nnd2;

            fxy = fieldxy + 2*j;

            logverb("Field object %i: (%g, %g)\n", j, fxy[0], fxy[1]);

            nn = kdtree_nearest_neighbour(itree, fxy, &nnd2);
            logverb("  Nearest index star is %g pixels away.\n",
                    sqrt(nnd2));

            res = kdtree_rangesearch_options(itree, fxy, pixr2 * nsigma*nsigma, KD_OPTIONS_SMALL_RADIUS);
            if (!res || !res->nres) {
                logverb("  No index stars within %g pixels\n", sqrt(pixr2)*nsigma);
                continue;
            }
            logverb("  %i index stars within %g pixels\n", res->nres, sqrt(pixr2)*nsigma);

            /*
             for (k=0; k<res->nres; k++) {
             intmap_append(corr_f2i, j, il_access(starlist, res->inds[k]));
             }
             */
            //intmap_append(corr_f2i, res->inds[k], il_access(starlist, j));
            kdtree_free_query(res);
        }


        if (log_get_level() >= LOG_VERB) {
            // See what quads could be built from the index stars with correspondences.
            double* corrxy;
            int k, N = il_size(indstarswithcorrs);
            corrxy = malloc(N * 2 * sizeof(double));
            for (j=0; j<N; j++) {
                int ind = il_get(indstarswithcorrs, j);
                corrxy[2*j+0] = dl_get(starxylist, ind*2+0);
                corrxy[2*j+1] = dl_get(starxylist, ind*2+1);
            }
            for (j=0; j<N; j++) {
                for (k=0; k<j; k++) {
                    int m;
                    double q2 = distsq(corrxy + j*2, corrxy + k*2, 2);
                    double qc[2];
                    int nvalid;
                    qc[0] = (corrxy[j*2+0] + corrxy[k*2+0]) / 2.0;
                    qc[1] = (corrxy[j*2+1] + corrxy[k*2+1]) / 2.0;
                    nvalid = 0;
                    for (m=0; m<N; m++) {
                        if (m == j || m == k)
                            continue;
                        if (distsq(qc, corrxy + m*2, 2) < q2/4.0)
                            nvalid++;
                    }
                    logverb("  Quad diameter: %g pix (%g arcmin): %i stars in the circle.\n",
                            sqrt(q2), sqrt(q2) * wcsscale / 60.0, nvalid);
                }
            }
            free(corrxy);
        }

        uniqquadlist = il_new(16);
        quadlist = il_new(16);

        // For each index star, find the quads of which it is part.
        for (j=0; j<il_size(starlist); j++) {
            int k;
            int starnum = il_get(starlist, j);
            if (qidxfile_get_quads(qidx, starnum, &quads, &nquads)) {
                logmsg("Failed to get quads for star %i.\n", starnum);
                exit(-1);
            }
            //logmsg("star %i is involved in %i quads.\n", starnum, nquads);
            for (k=0; k<nquads; k++) {
                il_insert_ascending(quadlist, quads[k]);
                il_insert_unique_ascending(uniqquadlist, quads[k]);

                il_append(istars[j].allquads, quads[k]);
            }
        }
        logmsg("Found %i quads partially contained in the field.\n", (int)il_size(uniqquadlist));

        dimquads = quadfile_dimquads(indx->quads);
        dimcodes = dimquad2dimcode(dimquads);

        // Find quads that are fully contained in the image.
        fullquadlist = il_new(16);
        for (j=0; j<il_size(uniqquadlist); j++) {
            int quad = il_get(uniqquadlist, j);
            int ind = il_index_of(quadlist, quad);
            if (log_get_level() >= LOG_VERB) {
                int k, nin=0, ncorr=0;
                unsigned int stars[dimquads];
                for (k=0; k<dimquads; k++) {
                    if (ind+k >= il_size(quadlist))
                        break;
                    if (il_get(quadlist, ind+k) != quad)
                        break;
                    nin++;
                }
                quadfile_get_stars(indx->quads, quad, stars);
                for (k=0; k<dimquads; k++)
                    if (il_contains(corrstarlist, stars[k]))
                        ncorr++;
                debug("Quad %i has %i stars in the field (%i with correspondences).\n", quad, nin, ncorr);
            }
            if (ind + (dimquads-1) >= il_size(quadlist))
                continue;
            if (il_get(quadlist, ind+(dimquads-1)) != quad)
                continue;
            il_append(fullquadlist, quad);
        }
        logmsg("Found %i quads fully contained in the field.\n", (int)il_size(fullquadlist));


        {
            double* sortdata = startree_get_data_column(indx->starkd, "r", starinds, Ngood);
            int k;
            int* sweeps = malloc(Ngood * sizeof(int));
            int* perm = NULL;
            int* maxsweep = malloc(il_size(uniqquadlist) * sizeof(int));
            unsigned int stars[DQMAX];

            // DEBUG -- how does "sweep" correspond to column "r" ?
            // (answer: small sweep --> small r, pretty much.)
            for (k=0; k<Ngood; k++) {
                sweeps[k] = startree_get_sweep(indx->starkd, starinds[k]);
            }
            perm = permuted_sort(sweeps, sizeof(int), compare_ints_asc, NULL, Ngood);
            if (sortdata) {
                for (k=0; k<Ngood; k++) {
                    logverb("  sweep %i, r mag %g\n", sweeps[perm[k]], sortdata[perm[k]]);
                }
            }

            /*
             for (k=0; k<Ngood; k++) {
             int star = starinds[perm[k]];
             double sd = sortdata[perm[k]];
             il* lst;
             lst = (il*)intmap_find(corr_i2f, star, FALSE);
             logmsg("Index star %i (%i), sweep %i, mag %g.  %i correspondences\n", k, star, sweeps[perm[k]], sd, (lst ? il_size(lst) : 0));
             if (lst) {
             for (j=0; j<il_size(lst); j++) {
             logmsg("  field star %i\n", il_get(lst, j));
             }
             }
             }
             */


            free(perm);


            for (j=0; j<il_size(uniqquadlist); j++) {
                int quad = il_get(uniqquadlist, j);
                int ms = 0;
                quadfile_get_stars(indx->quads, quad, stars);
                for (k=0; k<dimquads; k++) {
                    int sweep = startree_get_sweep(indx->starkd, stars[k]);
                    ms = MAX(ms, sweep);
                }
                maxsweep[j] = ms;
            }
            perm = permuted_sort(maxsweep, sizeof(int), compare_ints_asc, NULL, il_size(uniqquadlist));

            logverb("\nQuads completely within the image:\n");
            for (j=0; j<il_size(uniqquadlist); j++) {
                int quad = il_get(uniqquadlist, perm[j]);
                if (!il_contains(fullquadlist, quad))
                    continue;
                quadfile_get_stars(indx->quads, quad, stars);
                logverb("(full) quad %i: made from stars with sweeps:", quad);
                for (k=0; k<dimquads; k++) {
                    int sweep = startree_get_sweep(indx->starkd, stars[k]);
                    logverb(" %i", sweep);
                }
                logverb("\n");
            }

            logverb("\nNearby quads, not completely within the image:\n");
            for (j=0; j<il_size(uniqquadlist); j++) {
                int quad = il_get(uniqquadlist, perm[j]);
                if (il_contains(fullquadlist, quad))
                    continue;
                quadfile_get_stars(indx->quads, quad, stars);
                logverb("(near) quad %i: made from stars with sweeps:", quad);
                for (k=0; k<dimquads; k++) {
                    int sweep = startree_get_sweep(indx->starkd, stars[k]);
                    logverb(" %i", sweep);
                }
                logverb("\n");
            }

            free(perm);
            free(maxsweep);

            free(sweeps);
            free(sortdata);
        }



        if (quadplotfn) {
            for (j=0; j<il_size(fullquadlist); j++) {
                char* fn;
                int SW, SH;
                int k;
                plot_args_t* pargs;
                plotxy_t* pxy;
                plotindex_t* pind;
                int quad;
                unsigned int stars[dimquads];
                double quadxy[DQMAX * 2];

                asprintf(&fn, quadplotfn, j);
                SW = (int)(quadplotscale * W);
                SH = (int)(quadplotscale * H);

                pargs = plotstuff_new();
                pargs->outformat = PLOTSTUFF_FORMAT_PNG;
                pargs->outfn = fn;
                plotstuff_set_size(pargs, SW, SH);

                plotstuff_set_color(pargs, "black");
                plotstuff_plot_layer(pargs, "fill");
                pargs->lw = 2.0;

                pxy = plot_xy_get(pargs);
                pxy->scale = quadplotscale;

                plotstuff_set_color(pargs, "red");
                plotstuff_set_markersize(pargs, 6);
                for (k=0; k<Nfield; k++)
                    plot_xy_vals(pxy, fieldxy[2*k+0], fieldxy[2*k+1]);
                plotstuff_plot_layer(pargs, "xy");

                plotstuff_set_color(pargs, "green");
                //plotstuff_set_marker(pargs, "crosshair");
                plotstuff_set_markersize(pargs, 4);
                for (k=0; k<dl_size(starxylist)/2; k++)
                    plot_xy_vals(pxy, dl_get(starxylist, 2*k+0), dl_get(starxylist, 2*k+1));
                plotstuff_plot_layer(pargs, "xy");

                pind = plot_index_get(pargs);

                quad = il_get(fullquadlist, j);
                quadfile_get_stars(indx->quads, quad, stars);
                for (k=0; k<dimquads; k++) {
                    int m = il_index_of(starlist, stars[k]);
                    quadxy[k*2+0] = dl_get(starxylist, 2*m+0) * quadplotscale;
                    quadxy[k*2+1] = dl_get(starxylist, 2*m+1) * quadplotscale;
                }

                plot_quad_xy(pargs->cairo, quadxy, dimquads);
                cairo_stroke(pargs->cairo);

                plotstuff_output(pargs);

                plotstuff_free(pargs);
                logmsg("Wrote %s\n", fn);
                free(fn);
            }
        }


        // Find the stars that are in quads.
        starsinquadslist = il_new(16);
        for (j=0; j<il_size(uniqquadlist); j++) {
            int k;
            unsigned int stars[dimquads];
            int quad = il_get(uniqquadlist, j);
            quadfile_get_stars(indx->quads, quad, stars);
            for (k=0; k<dimquads; k++)
                il_insert_unique_ascending(starsinquadslist, stars[k]);
        }
        logmsg("Found %i index stars involved in quads (with at least one star contained in the image).\n", (int)il_size(starsinquadslist));

        // Find the stars that are in quads that are completely contained.
        starsinquadsfull = il_new(16);
        for (j=0; j<il_size(fullquadlist); j++) {
            int k;
            unsigned int stars[dimquads];
            int quad = il_get(fullquadlist, j);
            quadfile_get_stars(indx->quads, quad, stars);
            for (k=0; k<dimquads; k++)
                il_insert_unique_ascending(starsinquadsfull, stars[k]);
        }
        logmsg("Found %i index stars involved in quads (with all stars contained in the image).\n", (int)il_size(starsinquadsfull));

        // For each index object involved in quads, search for a correspondence.
        corrstars = il_new(16);
        corrfield = il_new(16);
        for (j=0; j<il_size(starsinquadslist); j++) {
            int star;
            double sxyz[3];
            double sxy[2];
            kdtree_qres_t* fres;
            star = il_get(starsinquadslist, j);
            if (startree_get(indx->starkd, star, sxyz)) {
                logmsg("Failed to get position for star %i.\n", star);
                exit(-1);
            }
            if (!sip_xyzarr2pixelxy(&sip, sxyz, sxy, sxy+1)) {
                logmsg("SIP backward for star %i.\n", star);
                exit(-1);
            }
            fres = kdtree_rangesearch_options(ftree, sxy, pixr2 * nsigma*nsigma,
                                              KD_OPTIONS_SMALL_RADIUS | KD_OPTIONS_SORT_DISTS);
            if (!fres || !fres->nres)
                continue;
            if (fres->nres > 1) {
                logmsg("%i matches for star %i.\n", fres->nres, star);
            }

            il_append(corrstars, star);
            il_append(corrfield, fres->inds[0]); //kdtree_permute(ftree, fres->inds[0]));

            logverb("  star %i: dist %g to field star %i\n", star, sqrt(fres->sdists[0]), fres->inds[0]);

            /*{
             double fx, fy;
             int fi;
             fi = il_get(corrfield, il_size(corrfield)-1);
             fx = fieldxy[2*fi + 0];
             fy = fieldxy[2*fi + 1];
             logmsg("star   %g,%g\n", sxy[0], sxy[1]);
             logmsg("field  %g,%g\n", fx, fy);
             }*/
        }
        logmsg("Found %i correspondences for stars involved in quads (with at least one star in the field).\n",
               (int)il_size(corrstars));

        // Find quads built only from stars with correspondences.
        corrquads = il_new(16);
        corruniqquads = il_new(16);
        for (j=0; j<il_size(corrstars); j++) {
            int k;
            int starnum = il_get(corrstars, j);
            if (qidxfile_get_quads(qidx, starnum, &quads, &nquads)) {
                logmsg("Failed to get quads for star %i.\n", starnum);
                exit(-1);
            }
            for (k=0; k<nquads; k++) {
                il_insert_ascending(corrquads, quads[k]);
                il_insert_unique_ascending(corruniqquads, quads[k]);
            }
        }

        // Find quads that are fully contained in the image.
        logverb("Looking at quads built from stars with correspondences...\n");
        corrfullquads = il_new(16);

        for (j=0; j<il_size(corruniqquads); j++) {
            int quad = il_get(corruniqquads, j);
            int ind = il_index_of(corrquads, quad);

            if (log_get_level() >= LOG_VERB) {
                int k, nin=0;
                for (k=0; k<dimquads; k++) {
                    if (ind+k >= il_size(corrquads))
                        break;
                    if (il_get(corrquads, ind+k) != quad)
                        break;
                    nin++;
                }
                debug("  Quad %i has %i stars with correspondences.\n", quad, nin);
            }

            if (ind + (dimquads-1) >= il_size(corrquads))
                continue;
            if (il_get(corrquads, ind+(dimquads-1)) != quad)
                continue;
            il_append(corrfullquads, quad);
        }
        logmsg("Found %i quads built from stars with correspondencs, fully contained in the field.\n", (int)il_size(corrfullquads));

        foundquads = bl_new(16, sizeof(foundquad_t));

        for (j=0; j<il_size(corrfullquads); j++) {
            unsigned int stars[dimquads];
            int k;
            int ind;
            double realcode[dimcodes];
            double fieldcode[dimcodes];
            tan_t wcs;
            MatchObj mo;
            foundquad_t fq;
            double codedist;

            int quad = il_get(corrfullquads, j);

            memset(&mo, 0, sizeof(MatchObj));

            quadfile_get_stars(indx->quads, quad, stars);

            codetree_get(indx->codekd, quad, realcode);

            for (k=0; k<dimquads; k++) {
                int find;
                // position of corresponding field star.
                ind = il_index_of(corrstars, stars[k]);
                assert(ind >= 0);
                find = il_get(corrfield, ind);
                mo.quadpix[k*2 + 0] = fieldxy[find*2 + 0];
                mo.quadpix[k*2 + 1] = fieldxy[find*2 + 1];
                // index star xyz.
                startree_get(indx->starkd, stars[k], mo.quadxyz + 3*k);

                mo.star[k] = stars[k];
                mo.field[k] = find;
            }

            logmsg("\nquad #%i (quad id %i): stars", j, quad);
            for (k=0; k<dimquads; k++)
                logmsg(" %i", mo.field[k]);
            logmsg("\n");

            codefile_compute_field_code(mo.quadpix, fieldcode, dimquads);

            //logmsg(" code (index): %.3f,%.3f,%.3f,%.3f\n", realcode[0], realcode[1], realcode[2], realcode[3]);
            //logmsg(" code (field): %.3f,%.3f,%.3f,%.3f\n", fieldcode[0], fieldcode[1], fieldcode[2], fieldcode[3]);

            quad_enforce_invariants(mo.star, realcode, dimquads, dimcodes);
            quad_enforce_invariants(mo.field, fieldcode, dimquads, dimcodes);
            codedist = sqrt(distsq(realcode, fieldcode, dimcodes));
            logmsg("  code distance (normal parity): %g\n", codedist);

            quad_flip_parity(fieldcode, fieldcode, dimcodes);
            quad_enforce_invariants(mo.star, realcode, dimquads, dimcodes);
            quad_enforce_invariants(mo.field, fieldcode, dimquads, dimcodes);
            codedist = sqrt(distsq(realcode, fieldcode, dimcodes));
            logmsg("  code distance (flip parity): %g\n", codedist);

            fit_tan_wcs(mo.quadxyz, mo.quadpix, dimquads, &wcs, NULL);
            wcs.imagew = W;
            wcs.imageh = H;

            {
                double pscale = tan_pixel_scale(&wcs);
                logmsg("  quad scale: %g arcsec/pix -> field size %g x %g arcmin\n",
                       pscale, arcsec2arcmin(pscale * W), arcsec2arcmin(pscale * H));
            }

            logverb("Distances between corresponding stars:\n");
            for (k=0; k<il_size(corrstars); k++) {
                int star = il_get(corrstars, k);
                int field = il_get(corrfield, k);
                double* fxy;
                double xyz[2];
                double sxy[2];
                double d;
                anbool ok;

                startree_get(indx->starkd, star, xyz);
                ok = tan_xyzarr2pixelxy(&wcs, xyz, sxy, sxy+1);
                fxy = fieldxy + 2*field;
                d = sqrt(distsq(fxy, sxy, 2));

                logverb("  correspondence: field star %i: distance %g pix\n", field, d);
            }

            logmsg("  running verify() with the found WCS:\n");

            //log_set_level(LOG_ALL);
            log_set_level(log_get_level() + 1);

            {
                double llxyz[3];
                verify_field_t* vf;
                double verpix2 = pixr2;

                tan_pixelxy2xyzarr(&wcs, W/2.0, H/2.0, mo.center);
                tan_pixelxy2xyzarr(&wcs, 0, 0, llxyz);
                mo.radius = sqrt(distsq(mo.center, llxyz, 3));
                mo.radius_deg = dist2deg(mo.radius);
                mo.scale = tan_pixel_scale(&wcs);
                mo.dimquads = dimquads;
                memcpy(&mo.wcstan, &wcs, sizeof(tan_t));
                mo.wcs_valid = TRUE;

                vf = verify_field_preprocess(xy);

                verify_hit(indx->starkd, indx->cutnside, &mo, NULL, vf, verpix2,
                           DEFAULT_DISTRACTOR_RATIO, W, H,
                           log(1e-100), log(1e9), LARGE_VAL, TRUE, FALSE);

                verify_field_free(vf);
            }

            log_set_level(loglvl);

            logmsg("Verify log-odds %g (odds %g)\n", mo.logodds, exp(mo.logodds));


            memset(&fq, 0, sizeof(foundquad_t));
            //memcpy(fq.stars, stars, dimquads);
            fq.codedist = codedist;
            fq.logodds = mo.logodds;
            fq.pscale = tan_pixel_scale(&wcs);
            fq.quadnum = quad;
            memcpy(&(fq.mo), &mo, sizeof(MatchObj));
            bl_append(foundquads, &fq);
        }

        // Sort the found quads by star index...
        bl_sort(foundquads, sort_fq_by_stars);

        logmsg("\n\n\n");
        logmsg("Sorted by star number (ie, the order they'll be found):\n\n");
        int ngood = 0;
        double maxcodedist = 0.01;
        double minlogodds = log(1e9);
        logmsg("Assuming max code distance %g and min logodds %g\n", maxcodedist, minlogodds);

        for (j=0; j<bl_size(foundquads); j++) {
            int k;
            foundquad_t* fq = bl_access(foundquads, j);
            if (fq->codedist <= maxcodedist && fq->logodds >= minlogodds) {
                ngood++;
                logmsg("*** ");
            }
            logmsg("quad #%i: stars", fq->quadnum);
            for (k=0; k<fq->mo.dimquads; k++) {
                logmsg(" %i", fq->mo.field[k]);
            }
            logmsg("\n");
            logmsg("  codedist %g\n", fq->codedist);
            logmsg("  logodds %g (odds %g)\n", fq->logodds, exp(fq->logodds));
        }
        printf("\nTotal of %i quads that would solve the image.\n", ngood);



        if (objplotfn) {
            il* closeistars = il_new(256);
            int SW, SH;
            plot_args_t* pargs;
            sip_t scalesip;

            SW = (int)(quadplotscale * W);
            SH = (int)(quadplotscale * H);

            pargs = plotstuff_new();
            pargs->outformat = PLOTSTUFF_FORMAT_PNG;
            plotstuff_set_size(pargs, SW, SH);

            sip_copy(&scalesip, &sip);
            tan_transform(&scalesip.wcstan, &scalesip.wcstan,
                          1, scalesip.wcstan.imagew,
                          1, scalesip.wcstan.imageh,
                          quadplotscale);
            plotstuff_set_wcs_sip(pargs, &scalesip);

            for (j=0; j<Nfield; j++) {
                char* fn;
                int k,m;
                plotxy_t* pxy;
                plotindex_t* pind;
                int quad;
                unsigned int stars[dimquads];
                //double quadxy[DQMAX * 2];
                double* fxy;
                int nn;
                double nnd2;
                kdtree_qres_t* res;
                double x,y;
                int closestart;

                fxy = fieldxy + 2*j;

                logverb("Field object %i: (%g, %g)\n", j, fxy[0], fxy[1]);

                nn = kdtree_nearest_neighbour(itree, fxy, &nnd2);
                logverb("  Nearest index star is %g pixels away.\n",
                        sqrt(nnd2));

                res = kdtree_rangesearch_options(itree, fxy, pixr2 * nsigma*nsigma, KD_OPTIONS_SMALL_RADIUS);
                if (!res || !res->nres) {
                    logverb("  No index stars within %g pixels\n", sqrt(pixr2)*nsigma);
                    continue;
                }

                logverb("  %i index stars within %g pixels\n", res->nres, sqrt(pixr2)*nsigma);

                // NOTE, "closeistars" contains indices in "starlist" / "starxy" directly.

                closestart = il_size(closeistars);

                for (k=0; k<res->nres; k++) {
                    il_append(closeistars, res->inds[k]);
                }

                asprintf(&fn, objplotfn, j);
                pargs->outfn = fn;

                plotstuff_set_color(pargs, "black");
                plotstuff_plot_layer(pargs, "fill");
                pargs->lw = 2.0;

                pxy = plot_xy_get(pargs);
                pxy->scale = quadplotscale;

                plotstuff_set_color(pargs, "green");
                //plotstuff_set_marker(pargs, "crosshair");

                plotstuff_set_markersize(pargs, 5);
                for (k=0; k<il_size(closeistars); k++) {
                    int ind = il_get(closeistars, k);
                    double x,y;
                    assert(ind >= 0);
                    x = dl_get(starxylist, 2*ind+0);
                    y = dl_get(starxylist, 2*ind+1);
                    plot_xy_vals(pxy, x, y);
                }
                plotstuff_plot_layer(pargs, "xy");
                plot_xy_clear_list(pxy);

                plotstuff_set_color(pargs, "darkgreen");
                plotstuff_set_markersize(pargs, 2);
                for (k=0; k<il_size(starlist); k++) {
                    //int star = il_get(starlist, k);
                    //if (il_contains(closeistars, star))
                    if (il_contains(closeistars, k))
                        continue;
                    x = dl_get(starxylist, 2*k+0);
                    y = dl_get(starxylist, 2*k+1);
                    plot_xy_vals(pxy, x, y);
                }
                plotstuff_plot_layer(pargs, "xy");
                plot_xy_clear_list(pxy);

                pind = plot_index_get(pargs);

                // All quads incident on index stars near field stars we've looked at (cumulative)
                for (k=0; k<il_size(uniqquadlist); k++) {
                    int nclose = 0;
                    anbool thistime = FALSE;
                    quad = il_get(uniqquadlist, k);
                    quadfile_get_stars(indx->quads, quad, stars);
                    for (m=0; m<dimquads; m++) {
                        int cind;
                        int ind = il_index_of(starlist, stars[m]);
                        if (ind == -1)
                            continue;
                        cind = il_index_of(closeistars, ind);
                        if (cind < 0)
                            continue;
                        nclose++;
                        if (cind >= closestart)
                            thistime = TRUE;
                        //logverb("quad %i: star %i, starlist ind %i, close ind %i (start %i)\n", quad, stars[m], ind, cind, closestart);
                    }
                    if (!nclose)
                        continue;

                    /*
                     for (m=0; m<dimquads; m++) {
                     int ind = il_index_of(starlist, stars[m]);
                     assert(ind >= 0);
                     quadxy[m*2+0] = dl_get(starxylist, 2*ind+0) * quadplotscale;
                     quadxy[m*2+1] = dl_get(starxylist, 2*ind+1) * quadplotscale;
                     }
                     plot_quad_xy(pargs->cairo, quadxy, dimquads);
                     */

                    if (thistime)
                        logverb("Quad %i: made from %i stars we've seen so far.\n", quad, nclose);

                    if (thistime) {
                        //plotstuff_set_color(pargs, "green");
                        cairo_set_color(pargs->cairo, "green");
                    } else {
                        //plotstuff_set_color(pargs, "darkgreen");
                        cairo_set_color(pargs->cairo, "darkgreen");
                    }

                    plot_index_plotquad(pargs->cairo, pargs, pind, indx, quad, dimquads);
                }


                plotstuff_set_color(pargs, "red");
                plotstuff_set_markersize(pargs, 6);
                for (k=0; k<=j; k++)
                    plot_xy_vals(pxy, fieldxy[2*k+0], fieldxy[2*k+1]);
                plotstuff_plot_layer(pargs, "xy");
                plot_xy_clear_list(pxy);

                plotstuff_set_markersize(pargs, 10);
                plot_xy_vals(pxy, fxy[0], fxy[1]);
                plotstuff_plot_layer(pargs, "xy");
                plot_xy_clear_list(pxy);

                plotstuff_set_color(pargs, "darkred");
                plotstuff_set_markersize(pargs, 3);
                for (k=j+1; k<Nfield; k++)
                    plot_xy_vals(pxy, fieldxy[2*k+0], fieldxy[2*k+1]);
                plotstuff_plot_layer(pargs, "xy");
                plot_xy_clear_list(pxy);

                plotstuff_output(pargs);

                logmsg("Wrote %s\n", fn);
                free(fn);

                kdtree_free_query(res);
            }

            plotstuff_free(pargs);
            free(itree->data.any);
            kdtree_free(itree);
        }

        il_free(fullquadlist);
        il_free(uniqquadlist);
        il_free(quadlist);
        il_free(starlist);
        il_free(corrstarlist);
    }

    if (xylist_close(xyls)) {
        logmsg("Failed to close XYLS file.\n");
    }
    return 0;
}
