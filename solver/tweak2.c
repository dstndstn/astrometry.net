/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <stdio.h>
#include <assert.h>

#include "os-features.h"
#include "bl.h"
#include "fit-wcs.h"
#include "sip.h"
#include "sip_qfits.h"
#include "sip-utils.h"
#include "scamp.h"
#include "log.h"
#include "errors.h"
#include "tweak.h"
#include "matchfile.h"
#include "matchobj.h"
#include "boilerplate.h"
#include "xylist.h"
#include "rdlist.h"
#include "mathutil.h"
#include "verify.h"
#include "fitsioutils.h"


// Tweak debug plots?
#define TWEAK_DEBUG_PLOTS 0
#if TWEAK_DEBUG_PLOTS
#include "plotstuff.h"
#include "plotimage.h"
#include "cairoutils.h"

static void makeplot(const char* plotfn, char* bgimgfn, int W, int H,
                     int Nfield, double* fieldpix, double* fieldsigma2s,
                     int Nindex, double* indexpix, int besti, int* theta,
                     double* crpix, int* testperm,
                     double * qc) {
    int i;
    plot_args_t pargs;
    plotimage_t* img;
    cairo_t* cairo;
    int ti;
    logmsg("Creating plot %s\n", plotfn);
    plotstuff_init(&pargs);
    pargs.outformat = PLOTSTUFF_FORMAT_PNG;
    pargs.outfn = plotfn;
    pargs.fontsize = 12;
    if (bgimgfn) {
        img = plotstuff_get_config(&pargs, "image");
        img->format = PLOTSTUFF_FORMAT_JPG;
        plot_image_set_filename(img, bgimgfn);
        plot_image_setsize(&pargs, img);
        plotstuff_run_command(&pargs, "image");
    } else {
        float rgba[4] = {0, 0, 0.1, 1.0};
        plotstuff_set_size(&pargs, W, H);
        //plotstuff_init2(&pargs);
        plotstuff_set_rgba(&pargs, rgba);
        plotstuff_run_command(&pargs, "fill");
    }
    cairo = pargs.cairo;
    // red circles around every field star.
    cairo_set_color(cairo, "gray");
    for (i=0; i<Nfield; i++) {
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                               fieldpix[2*i+0], fieldpix[2*i+1],
                               2.0 * sqrt(fieldsigma2s[i]));
        cairo_stroke(cairo);
    }
    // green crosshairs at every index star.
    cairo_set_color(cairo, "green");
    for (i=0; i<Nindex; i++) {
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_XCROSSHAIR,
                               indexpix[2*i+0], indexpix[2*i+1], 3);
        cairo_stroke(cairo);
    }
    // thick white circles for corresponding field stars.
    cairo_set_line_width(cairo, 2);
    for (ti=0; ti<=besti; ti++) {
        if (testperm)
            i = testperm[ti];
        else
            i = ti;
        //printf("field %i -> index %i\n", i, theta[i]);
        if (theta[i] < 0)
            continue;
        cairo_set_color(cairo, "white");
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                               fieldpix[2*i+0], fieldpix[2*i+1],
                               2.0 * sqrt(fieldsigma2s[i]));
        cairo_stroke(cairo);
        // thick cyan crosshairs for corresponding index stars.
        cairo_set_color(cairo, "cyan");
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_XCROSSHAIR,
                               indexpix[2*theta[i]+0],
                               indexpix[2*theta[i]+1],
                               3);
        cairo_stroke(cairo);
    }

    cairo_set_line_width(cairo, 2);

    //for (i=0; i<=besti; i++) {
    for (ti=0; ti<Nfield; ti++) {
        anbool mark = TRUE;
        if (testperm)
            i = testperm[ti];
        else
            i = ti;
        switch (theta[i]) {
        case THETA_DISTRACTOR:
            cairo_set_color(cairo, "red");
            break;
        case THETA_CONFLICT:
            cairo_set_color(cairo, "yellow");
            break;
        case THETA_FILTERED:
            cairo_set_color(cairo, "orange");
            break;
        default:
            if (theta[i] < 0) {
                cairo_set_color(cairo, "gray");
            } else {
                cairo_set_color(cairo, "white");
            }
            mark = FALSE;
        }

        if (mark) {
            cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CIRCLE,
                                   fieldpix[2*i+0], fieldpix[2*i+1],
                                   2.0 * sqrt(fieldsigma2s[i]));
            cairo_stroke(cairo);
        }

        if (ti <= MAX(besti, 10)) {
            char label[32];
            sprintf(label, "%i", i);
            plotstuff_text_xy(&pargs, fieldpix[2*i+0], fieldpix[2*i+1], label);
        }
        if (i == besti) {
            cairo_set_line_width(cairo, 1);
        }
    }


    if (crpix) {
        cairo_set_color(cairo, "yellow");
        cairo_set_line_width(cairo, 4);
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CROSSHAIR,
                               crpix[0], crpix[1], 10);
        cairo_stroke(cairo);
    }

    if (qc) {
        cairo_set_color(cairo, "skyblue");
        cairo_set_line_width(cairo, 4);
        cairoutils_draw_marker(cairo, CAIROUTIL_MARKER_CROSSHAIR,
                               qc[0], qc[1], 10);
        cairo_stroke(cairo);
    }

    plotstuff_output(&pargs);
    logmsg("Wrote plot %s\n", plotfn);
}

static char* tdebugfn(const char* name) {
    static char fn[256];
    static int plotnum = 0;
    sprintf(fn, "tweak-%03i-%s.png", plotnum, name);
    plotnum++;
    return fn;
}

#define TWEAK_DEBUG_PLOT(name, W, H, Nfield, fieldxy, fieldsig2,        \
                         Nindex, indexxy, besti, theta, crpix, testperm, qc) \
    makeplot(tdebugfn(name), NULL, W, H, Nfield, fieldxy, fieldsig2,	\
             Nindex, indexxy, besti, theta, crpix, testperm, qc);


#else

#define TWEAK_DEBUG_PLOT(name, W, H, Nfield, fieldxy, fieldsig2,        \
                         Nindex, indexxy, besti, theta, crpix, testperm, qc) \
    do{}while(0)

#endif





sip_t* tweak2(const double* fieldxy, int Nfield,
              double fieldjitter,
              int W, int H,
              const double* indexradec, int Nindex,
              double indexjitter,
              const double* quadcenter, double quadR2,
              double distractors,
              double logodds_bail,
              int sip_order,
              int sip_invorder,
              const sip_t* startwcs,
              sip_t* destwcs,
              int** newtheta, double** newodds,
              double* crpix,
              double* p_logodds,
              int* p_besti,
              int* testperm,
              int startorder) {
    int order;
    sip_t* sipout;
    int* indexin;
    double* indexpix;
    double* fieldsigma2s;
    double* weights;
    double* matchxyz;
    double* matchxy;
    int i, Nin=0;
    double logodds = 0;
    int besti = -1;
    int* theta = NULL;
    double* odds = NULL;
    int* refperm = NULL;
    double qc[2];

    memcpy(qc, quadcenter, 2*sizeof(double));

    if (destwcs)
        sipout = destwcs;
    else
        sipout = sip_create();

    indexin = malloc(Nindex * sizeof(int));
    indexpix = malloc(2 * Nindex * sizeof(double));
    fieldsigma2s = malloc(Nfield * sizeof(double));
    weights = malloc(Nfield * sizeof(double));
    matchxyz = malloc(Nfield * 3 * sizeof(double));
    matchxy = malloc(Nfield * 2 * sizeof(double));

    // FIXME --- hmmm, how do the annealing steps and iterating up to
    // higher orders interact?

    assert(startwcs);
    memcpy(sipout, startwcs, sizeof(sip_t));

    logverb("tweak2: starting orders %i, %i\n", sipout->a_order, sipout->ap_order);

    if (!sipout->wcstan.imagew)
        sipout->wcstan.imagew = W;
    if (!sipout->wcstan.imageh)
        sipout->wcstan.imageh = H;

    logverb("Tweak2: starting from WCS:\n");
    if (log_get_level() >= LOG_VERB)
        sip_print_to(sipout, stdout);

    for (order=startorder; order <= sip_order; order++) {
        int step;
        int STEPS = 100;
        // variance growth rate wrt radius.
        double gamma = 1.0;
        //logverb("Starting tweak2 order=%i\n", order);

        for (step=0; step<STEPS; step++) {
            double iscale;
            double ijitter;
            double ra, dec;
            double R2;
            int Nmatch;
            int nmatch, nconf, ndist;
            double pix2;
            double totalweight;

            // clean up from last round (we do it here so that they're
            // valid when we leave the loop)
            free(theta);
            free(odds);
            free(refperm);

            // Anneal
            gamma = pow(0.9, step);
            if (step == STEPS-1)
                gamma = 0.0;
            logverb("Annealing: order %i, step %i, gamma = %g\n", order, step, gamma);
			
            debug("Using input WCS:\n");
            if (log_get_level() > LOG_VERB)
                sip_print_to(sipout, stdout);

            // Project reference sources into pixel space; keep the ones inside image bounds.
            Nin = 0;
            for (i=0; i<Nindex; i++) {
                anbool ok;
                double x,y;
                ra  = indexradec[2*i + 0];
                dec = indexradec[2*i + 1];
                ok = sip_radec2pixelxy(sipout, ra, dec, &x, &y);
                if (!ok)
                    continue;
                if (!sip_pixel_is_inside_image(sipout, x, y))
                    continue;
                indexpix[Nin*2+0] = x;
                indexpix[Nin*2+1] = y;
                indexin[Nin] = i;
                Nin++;
            }
            logverb("%i reference sources within the image.\n", Nin);
            //logverb("CRPIX is (%g,%g)\n", sip.wcstan.crpix[0], sip.wcstan.crpix[1]);

            if (Nin == 0) {
                sip_free(sipout);
                free(matchxy);
                free(matchxyz);
                free(weights);
                free(fieldsigma2s);
                free(indexpix);
                free(indexin);
                return NULL;
            }

            iscale = sip_pixel_scale(sipout);
            ijitter = indexjitter / iscale;
            //logverb("With pixel scale of %g arcsec/pixel, index adds jitter of %g pix.\n", iscale, ijitter);

            /* CHECK
             for (i=0; i<Nin; i++) {
             double x,y;
             int ii = indexin[i];
             sip_radec2pixelxy(sipout, indexradec[2*ii+0], indexradec[2*ii+1], &x, &y);
             logverb("indexin[%i]=%i; (%.1f,%.1f) -- (%.1f,%.1f)\n",
             i, ii, indexpix[i*2+0], indexpix[i*2+1], x, y);
             }
             */

            for (i=0; i<Nfield; i++) {
                R2 = distsq(qc, fieldxy + 2*i, 2);
                fieldsigma2s[i] = (square(fieldjitter) + square(ijitter)) * (1.0 + gamma * R2/quadR2);
            }

            if (order == 1 && step == 0 && TWEAK_DEBUG_PLOTS) {
                TWEAK_DEBUG_PLOT("init", W, H, Nfield, fieldxy, fieldsigma2s,
                                 Nin, indexpix, *p_besti, *newtheta,
                                 sipout->wcstan.crpix, testperm, qc);
            }

            /*
             logodds = verify_star_lists(indexpix, Nin,
             fieldxy, fieldsigma2s, Nfield,
             W*H, distractors,
             logodds_bail, LARGE_VAL,
             &besti, &odds, &theta, NULL,
             &testperm);
             */

            pix2 = square(fieldjitter);
            logodds = verify_star_lists_ror(indexpix, Nin,
                                            fieldxy, fieldsigma2s, Nfield,
                                            pix2, gamma, qc, quadR2,
                                            W, H, distractors,
                                            logodds_bail, LARGE_VAL,
                                            &besti, &odds, &theta, NULL,
                                            &testperm, &refperm);

            logverb("Logodds: %g\n", logodds);
            verify_count_hits(theta, besti, &nmatch, &nconf, &ndist);
            logverb("%i matches, %i distractors, %i conflicts (at best log-odds); %i field sources, %i index sources\n", nmatch, ndist, nconf, Nfield, Nin);
            verify_count_hits(theta, Nfield-1, &nmatch, &nconf, &ndist);
            logverb("%i matches, %i distractors, %i conflicts (all sources)\n", nmatch, ndist, nconf);
            if (log_get_level() >= LOG_VERB) {
                matchobj_log_hit_miss(theta, testperm, besti+1, Nfield, LOG_VERB, "Hit/miss: ");
            }

            /*
             logverb("\nAfter verify():\n");
             for (i=0; i<Nin; i++) {
             double x,y;
             int ii = indexin[refperm[i]];
             sip_radec2pixelxy(sipout, indexradec[2*ii+0], indexradec[2*ii+1], &x, &y);
             logverb("indexin[%i]=%i; (%.1f,%.1f) -- (%.1f,%.1f)\n",
             i, ii, indexpix[i*2+0], indexpix[i*2+1], x, y);
             }
             */

            if (TWEAK_DEBUG_PLOTS) {
                char name[32];
                sprintf(name, "o%is%02ipre", order, step);
                TWEAK_DEBUG_PLOT(name, W, H, Nfield, fieldxy, fieldsigma2s,
                                 Nin, indexpix, besti, theta,
                                 sipout->wcstan.crpix, testperm, qc);
            }

            Nmatch = 0;
            debug("Weights:");
            for (i=0; i<Nfield; i++) {
                double ra,dec;
                if (theta[i] < 0)
                    continue;
                assert(theta[i] < Nin);
                int ii = indexin[refperm[theta[i]]];
                assert(ii < Nindex);
                assert(ii >= 0);

                ra  = indexradec[ii*2+0];
                dec = indexradec[ii*2+1];
                radecdeg2xyzarr(ra, dec, matchxyz + Nmatch*3);
                memcpy(matchxy + Nmatch*2, fieldxy + i*2, 2*sizeof(double));
                weights[Nmatch] = verify_logodds_to_weight(odds[i]);
                debug(" %.2f", weights[Nmatch]);
                Nmatch++;

                /*
                 logverb("match img (%.1f,%.1f) -- ref (%.1f, %.1f), odds %g, wt %.3f\n",
                 fieldxy[i*2+0], fieldxy[i*2+1],
                 indexpix[theta[i]*2+0], indexpix[theta[i]*2+1],
                 odds[i],
                 weights[Nmatch-1]);
                 double xx,yy;
                 sip_radec2pixelxy(sipout, ra, dec, &xx, &yy);
                 logverb("check: (%.1f, %.1f)\n", xx, yy);
                 */
            }
            debug("\n");

            if (Nmatch < 2) {
                logverb("No matches -- aborting tweak attempt\n");
                free(theta);
                sip_free(sipout);
                free(matchxy);
                free(matchxyz);
                free(weights);
                free(fieldsigma2s);
                free(indexpix);
                free(indexin);
                return NULL;
            }

            // Update the "quad center" to be the weighted average matched star posn.
            qc[0] = qc[1] = 0.0;
            totalweight = 0.0;
            for (i=0; i<Nmatch; i++) {
                qc[0] += (weights[i] * matchxy[2*i+0]);
                qc[1] += (weights[i] * matchxy[2*i+1]);
                totalweight += weights[i];
            }
            qc[0] /= totalweight;
            qc[1] /= totalweight;
            logverb("Moved quad center to (%.1f, %.1f)\n", qc[0], qc[1]);

            //
            sipout->a_order = sipout->b_order = order;
            sipout->ap_order = sipout->bp_order = sip_invorder;
            logverb("tweak2: setting orders %i, %i\n", sipout->a_order, sipout->ap_order);

            if (crpix) {
                tan_t temptan;
                logverb("Moving tangent point to given CRPIX (%g,%g)\n", crpix[0], crpix[1]);
                fit_tan_wcs_move_tangent_point_weighted(matchxyz, matchxy, weights, Nmatch,
                                                        crpix, &sipout->wcstan, &temptan);
                fit_tan_wcs_move_tangent_point_weighted(matchxyz, matchxy, weights, Nmatch,
                                                        crpix, &temptan, &sipout->wcstan);
            }

            int doshift = 1;
            fit_sip_wcs(matchxyz, matchxy, weights, Nmatch,
                        &(sipout->wcstan), order, sip_invorder,
                        doshift, sipout);

            debug("Got SIP:\n");
            if (log_get_level() > LOG_VERB)
                sip_print_to(sipout, stdout);
            sipout->wcstan.imagew = W;
            sipout->wcstan.imageh = H;
        }
    }

    //logverb("Final logodds: %g\n", logodds);

    // Now, recompute final logodds after turning 'gamma' on again (?)
    // FIXME -- this counts the quad stars in the logodds...
    {
        double gamma = 1.0;
        double iscale;
        double ijitter;
        double ra, dec;
        double R2;
        int nmatch, nconf, ndist;
        double pix2;

        free(theta);
        free(odds);
        free(refperm);
        gamma = 1.0;
        // Project reference sources into pixel space; keep the ones inside image bounds.
        Nin = 0;
        for (i=0; i<Nindex; i++) {
            anbool ok;
            double x,y;
            ra  = indexradec[2*i + 0];
            dec = indexradec[2*i + 1];
            ok = sip_radec2pixelxy(sipout, ra, dec, &x, &y);
            if (!ok)
                continue;
            if (!sip_pixel_is_inside_image(sipout, x, y))
                continue;
            indexpix[Nin*2+0] = x;
            indexpix[Nin*2+1] = y;
            indexin[Nin] = i;
            Nin++;
        }
        logverb("%i reference sources within the image.\n", Nin);

        iscale = sip_pixel_scale(sipout);
        ijitter = indexjitter / iscale;
        for (i=0; i<Nfield; i++) {
            R2 = distsq(qc, fieldxy + 2*i, 2);
            fieldsigma2s[i] = (square(fieldjitter) + square(ijitter)) * (1.0 + gamma * R2/quadR2);
        }

        pix2 = square(fieldjitter);
        logodds = verify_star_lists_ror(indexpix, Nin,
                                        fieldxy, fieldsigma2s, Nfield,
                                        pix2, gamma, qc, quadR2,
                                        W, H, distractors,
                                        logodds_bail, LARGE_VAL,
                                        &besti, &odds, &theta, NULL,
                                        &testperm, &refperm);
        logverb("Logodds: %g\n", logodds);
        verify_count_hits(theta, besti, &nmatch, &nconf, &ndist);
        logverb("%i matches, %i distractors, %i conflicts (at best log-odds); %i field sources, %i index sources\n", nmatch, ndist, nconf, Nfield, Nin);
        verify_count_hits(theta, Nfield-1, &nmatch, &nconf, &ndist);
        logverb("%i matches, %i distractors, %i conflicts (all sources)\n", nmatch, ndist, nconf);
        if (log_get_level() >= LOG_VERB) {
            matchobj_log_hit_miss(theta, testperm, besti+1, Nfield, LOG_VERB,
                                  "Hit/miss: ");
        }

        if (TWEAK_DEBUG_PLOTS) {
            TWEAK_DEBUG_PLOT("final", W, H, Nfield, fieldxy, fieldsigma2s,
                             Nin, indexpix, besti, theta,
                             sipout->wcstan.crpix, testperm, qc);
        }
    }


    if (newtheta) {
        // undo the "indexpix" inside-image-bounds cut.
        (*newtheta) = malloc(Nfield * sizeof(int));
        for (i=0; i<Nfield; i++) {
            int nt;
            if (theta[i] < 0)
                nt = theta[i];
            else
                nt = indexin[refperm[theta[i]]];
            (*newtheta)[i] = nt;
        }
    }
    free(theta);
    free(refperm);

    if (newodds)
        *newodds = odds;
    else
        free(odds);

    logverb("Tweak2: final WCS:\n");
    if (log_get_level() >= LOG_VERB)
        sip_print_to(sipout, stdout);

    if (p_logodds)
        *p_logodds = logodds;
    if (p_besti)
        *p_besti = besti;

    free(indexin);
    free(indexpix);
    free(fieldsigma2s);
    free(weights);
    free(matchxyz);
    free(matchxy);

    return sipout;
}

