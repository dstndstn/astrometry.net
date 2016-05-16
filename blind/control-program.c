/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
 */

/**

 This is a sample telescope-control program.

 */

#include <unistd.h>
#include <stdio.h>
#include <getopt.h>
#include <libgen.h>
#include <math.h>

#include "os-features.h"
#include "engine.h"
#include "solver.h"
#include "index.h"
#include "starxy.h"
#include "matchobj.h"
#include "healpix.h"
#include "bl.h"
#include "log.h"
#include "errors.h"
#include "fileutils.h"

// required for this sample program, maybe not for yours...
#include "anqfits.h"
#include "image2xy.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "tic.h"

static const char* OPTIONS = "hvc:a:A:W:H:";

#define DEFAULT_IMAGEW 1024
#define DEFAULT_IMAGEH 1024
#define DEFAULT_ARCMIN_MIN 15.0
#define DEFAULT_ARCMIN_MAX 25.0

static void print_help(const char* progname) {
    printf("Usage:   %s\n"
           "   [-c <astrometry config file>]  (default: \"../etc/astrometry.cfg\" relative to this executable)\n"
           "   [-a <minimum field width>] in arcminutes, default %g\n"
           "   [-A <maximum field width>] in arcminutes, default %g\n"
           "   [-W <image width> ] in pixels, default %i\n"
           "   [-H <image height>] in pixels, default %i\n"
           "   [-v]: verbose\n"
           "\n"
           "    <FITS image filename> [<FITS image filename> ...]: the images to process.\n"
           "\n"
           "\n", progname, DEFAULT_ARCMIN_MIN, DEFAULT_ARCMIN_MAX,
           DEFAULT_IMAGEW, DEFAULT_IMAGEH);
}

// If you've got a TAN or SIP WCS, set "sip".  Otherwise, be sure to set
// (racenter, deccenter).
static void get_next_field(char* fits_image_fn,
                           int* nstars, double** starx, double** stary,
                           double** starflux, sip_t** sip,
                           double* ra, double* dec) {
    int i, N;
    simplexy_t simxy;
    // For this sample program I'm just going to read from a FITS image,
    // but you probably want to grab data fresh from the CCD...
    anqfits_t* anq;

    anq = anqfits_open(fits_image_fn);
    if (!anq) {
        ERROR("Failed to open FITS file \"%s\"\n", fits_image_fn);
        exit(-1);
    }

    simplexy_set_defaults(&simxy);
    simxy.image = anqfits_readpix(anq, 0, 0, 0, 0, 0, 0, PTYPE_FLOAT, NULL,
                                  &(simxy.nx), &(simxy.ny));
    anqfits_close(anq);
    if (!simxy.image) {
        ERROR("Failed to read image pixel from FITS file \"%s\"\n",
              fits_image_fn);
        exit(-1);
    }

    image2xy_run(&simxy, 0, 0);

    N = simxy.npeaks;
    *nstars = N;
    *starx = malloc(N * sizeof(double));
    *stary = malloc(N * sizeof(double));
    *starflux = malloc(N * sizeof(double));
    for (i=0; i<N; i++) {
        (*starx)[i] = simxy.x[i];
        (*stary)[i] = simxy.y[i];
        (*starflux)[i] = simxy.flux[i];
    }
    simplexy_free_contents(&simxy);
    // simplexy_free_contents frees the image data

    // Try reading SIP header...
    logmsg("Trying to read WCS header from %s...\n", fits_image_fn);
    errors_start_logging_to_string();
    *sip = sip_read_header_file(fits_image_fn, NULL);
    if (!*sip) {
        char* errmsg = errors_stop_logging_to_string("\n  ");
        logmsg("Reading WCS header failed:\n  %s\n", errmsg);
        free(errmsg);
    }
    // If that doesn't work, look for "RA" and "DEC" header cards.
    if (!*sip) {
        double infval = 1.0/0.0;
        qfits_header* hdr = anqfits_get_header2(fits_image_fn, 0);
        *ra = qfits_header_getdouble(hdr, "RA", infval);
        logverb("Looking for RA header (float): %g\n", *ra);
        if (isinf(*ra)) {
            char* rastr = qfits_header_getstr(hdr, "RA");
            logverb("Looking for RA header (string): >>%s<<\n", rastr);
            if (rastr) {
                *ra = atora(qfits_header_getstr(hdr, "RA"));
                logverb("Parsed to %g\n", *ra);
            }
        }
        *dec = qfits_header_getdouble(hdr, "DEC", infval);
        logverb("Looking for Dec header (float): %g\n", *dec);
        if (isinf(*dec)) {
            char* decstr = qfits_header_getstr(hdr, "DEC");
            logverb("Looking for Dec header (string): >>%s<<\n", decstr);
            if (decstr) {
                *dec = atora(qfits_header_getstr(hdr, "DEC"));
                logverb("Parsed to %g\n", *dec);
            }
        }
        qfits_header_destroy(hdr);
        logmsg("Using (RA,Dec) estimate (%g, %g)\n", *ra, *dec);
    }
}

/* Yuck: a struct required to hold data for the record_match_callback.
 Only required if you need the list of matched reference and image stars.
 */
struct callbackdata {
    solver_t* solver;
    MatchObj match;
};

/* This callback function is only required if you need the list of
 matched reference and image stars. */
static anbool record_match_callback(MatchObj* mo, void* userdata) {
    struct callbackdata* cb = userdata;
    MatchObj* mymatch = &(cb->match);
    solver_t* solver = cb->solver;
    int i;
    // copy "mo" to "mymatch"
    memcpy(mymatch, mo, sizeof(MatchObj));
	// steal these arrays from "mo": we memcpy'd the pointers above, now NULL
    // them out in "mo" to prevent them from being free'd.
	mo->theta = NULL;
	mo->matchodds = NULL;
	mo->refxyz = NULL;
	mo->refxy = NULL;
	mo->refstarid = NULL;
	mo->testperm = NULL;

    // Convert xyz to RA,Dec
    mymatch->refradec = malloc(mymatch->nindex * 2 * sizeof(double));
    for (i=0; i<mymatch->nindex; i++) {
        xyzarr2radecdegarr(mymatch->refxyz+i*3, mymatch->refradec+i*2);
    }
    mymatch->fieldxy = malloc(mymatch->nfield * 2 * sizeof(double));
    // whew! -- Copy the (permuted) image (field) stars.
    memcpy(mymatch->fieldxy, solver->vf->xy,
           mymatch->nfield * 2 * sizeof(double));

    // Accept this match.
    return TRUE;
}


int main(int argc, char** args) {
    char* configfn = NULL;
    int loglvl = LOG_MSG;

    // Image size in pixels.
    int imagew = DEFAULT_IMAGEW;
    int imageh = DEFAULT_IMAGEH;
    // Image angular width range, in arcminutes.
    double arcmin_width_min = DEFAULT_ARCMIN_MIN;
    double arcmin_width_max = DEFAULT_ARCMIN_MAX;

    int i, I, c;
    sl* imagefiles;

    engine_t* engine;
    solver_t* solver;
    double hprange;

    while ((c = getopt(argc, args, OPTIONS)) != -1)
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'v':
            loglvl++;
            break;
        case 'c':
            configfn = strdup(optarg);
            break;
        case 'W':
            imagew = atoi(optarg);
            break;
        case 'H':
            imageh = atoi(optarg);
            break;
        case 'a':
            arcmin_width_min = atof(optarg);
            break;
        case 'A':
            arcmin_width_max = atof(optarg);
            break;
        case '?':
        default:
            printf("Unknown flag %c\n", c);
            exit( -1);
        }

    imagefiles = sl_new(4);

    for (i=optind; i<argc; i++)
        sl_append(imagefiles, args[i]);

    if (!sl_size(imagefiles)) {
        printf("You must specify at least one FITS image file to read.\n");
        print_help(args[0]);
        exit(-1);
    }

    log_init(loglvl);

    // only required if you use qfits.
    fits_use_error_system();

    if (!configfn) {
        char *me, *mydir;
        me = find_executable(args[0], NULL);
        if (!me)
            me = strdup(args[0]);
        mydir = strdup(dirname(me));
        free(me);
        configfn = resolve_path("../etc/astrometry.cfg", mydir);
        free(mydir);
    }

    engine = engine_new();

    logmsg("Reading config file %s and loading indexes...\n", configfn);
    if (engine_parse_config_file(engine, configfn)) {
        logerr("Failed to parse (or encountered an error while interpreting) config file \"%s\"\n", configfn);
        exit( -1);
    }

    if (!pl_size(engine->indexes)) {
        logerr("You must list at least one index in the config file (%s)\n", configfn);
        exit( -1);
    }
    free(configfn);

    logmsg("Loaded %zu indexes.\n", pl_size(engine->indexes));

    // For a control program you almost certainly want to be using small enough
    // indexes that they fit in memory!
    // Maybe not -- maybe most of them won't be loaded because of
    // healpix constraints...
    if (!engine->inparallel) {
        size_t i;
        logerr("Forcing indexes_inparallel.\n");
        engine->inparallel = TRUE;
        // We can't just set "inparallel" because the index files are
        // loaded during config file parsing... must reload now.
        for (i=0; i<pl_size(engine->indexes); i++) {
            index_t* index = pl_get(engine->indexes, i);
            logmsg("Reloading index \"%s\"...\n", index->indexname);
            index_reload(index);
        }
    }

    // I assume that the engine config file only contains indexes that cover
    // the range of scales you are interested in.

    // Furthermore, I assume the range of scales is small enough so that if we
    // try to verify an existing WCS that claims the field is huge, we won't
    // accidentally try to load millions of stars.

    solver = solver_new();
    {
        double app_min, app_max;
        double qsf_min = 0.1;

        // compute scale range in arcseconds per pixel.
        app_min = arcmin2arcsec(arcmin_width_min / (double)imagew);
        app_max = arcmin2arcsec(arcmin_width_max / (double)imagew);
        solver->funits_lower = app_min;
        solver->funits_upper = app_max;

        // If you want to look at only a limited number of sources:
        // solver->endobj = 20;
        // Or you can limit the number of quads the solver tries:
        // solver->maxquads = 1000000;
        // solver->maxmatches = 1000000;

        // don't try teeny-tiny quads.
        solver->quadsize_min = qsf_min * MIN(imagew, imageh);

        // by determining whether your images have "positive" or
        // "negative" parity (sign of the determinant of WCS CD matrix),
        // you can set this and save half the compute time.
        // solver->parity = PARITY_NORMAL;
        // or
        // solver->parity = PARITY_FLIP;

        // This determines how good a match has to be.
        // (you can set it huge; most matches are overwhelmingly good)
        solver_set_keep_logodds(solver, log(1e12));

        // What is the radius of the bounding circle of a field?
        // (in units of distance on the unit sphere)
        // This is used to decide which indexes to use.
        // You could expand this to take into account the error you expect in
        // the initial WCS estimate.  However, currently the healpix code that
        // decides which healpixes are within range doesn't work if the range is
        // larger than the healpix side -- but that's probably 10s of degrees for
        // typical situations.
        hprange = arcsec2dist(app_max * hypot(imagew, imageh) / 2.0);
    }

    for (I=0;; I++) {
        size_t i, N;
        sip_t* sip;
        double racenter, deccenter;
        il* hplist = il_new(4);
        starxy_t* field;
        double *starx, *stary, *starflux;
        int nstars;
        double imagecx, imagecy;
        anbool solved = FALSE;
        char* img;
        double t0, t1;
        struct callbackdata cb;
        cb.solver = solver;
        MatchObj* match = &(cb.match);
        memset(match, 0, sizeof(MatchObj));

        // Get the next image...
        racenter = 0.0;
        deccenter = 0.0;
        img = sl_get(imagefiles, I % sl_size(imagefiles));
        logmsg("Reading image file %s\n", img);
        get_next_field(img, &nstars, &starx, &stary, &starflux, &sip,
                       &racenter, &deccenter);

        t0 = timenow();

        // Feed the image source coordinates to the solver...
        field = starxy_new(nstars, TRUE, FALSE);
        starxy_set_x_array(field, starx);
        starxy_set_y_array(field, stary);
        starxy_set_flux_array(field, starflux);
        starxy_sort_by_flux(field);
        solver_set_field(solver, field);
        free(starx);
        free(stary);
        free(starflux);

        // center of the image in pixels, according to FITS indexing.
        imagecx = (imagew - 1.0)/2.0;
        imagecy = (imageh - 1.0)/2.0;

        // Where is the center of the image according to the existing WCS?
        if (sip)
            sip_pixelxy2radec(sip, imagecx, imagecy, &racenter, &deccenter);

        // Which indexes should we use?  Use the WCS or RA,Dec estimate to decide.
        N = pl_size(engine->indexes);
        for (i=0; i<N; i++) {
            index_t* index = pl_get(engine->indexes, i);
            if (!isinf(racenter) &&
                !index_is_within_range(index, racenter, deccenter,
                                       dist2deg(hprange)))
                continue;
            logmsg("Adding index %s\n", index->indexname);
            solver_add_index(solver, index);
        }
        if (solver_n_indices(solver) == 0) {
            logmsg("No index files are within range of given RA,Dec center "
                   "and radius: (%g,%g), %g\n", racenter, deccenter,
                   dist2deg(hprange));
            goto skip;
        }

        // If you want the list of matched stars, you have to do this ugliness:
        solver->record_match_callback = record_match_callback;
        solver->userdata = &cb;

        if (sip) {
            logmsg("Trying to verify existing WCS...\n");
            solver_verify_sip_wcs(solver, sip);
            if (solver->best_match_solves) {
                // Existing WCS passed the test.
                logmsg("Existing WCS pass the verification test with odds ratio %g\n",
                       exp(solver->best_match.logodds));
                // the WCS is solver->best_match.wcstan
                solved = TRUE;
            } else {
                logmsg("Existing WCS failed the verification test...\n");
            }
        }

        if (!solved) {
            /*
             // Now, if you wanted to ignore the WCS and check all indexes, you
             // could do this:
             solver_clear_indexes(solver);
             for (i=0; i<N; i++) {
             index_t* index = pl_get(engine->indexes);
             solver_add_index(solver, index);
             }
             */
            // 
            solver->distance_from_quad_bonus = TRUE;

            solver_run(solver);

            if (solver->best_match_solves) {
                double ra, dec;
                double pscale;
                tan_t* wcs;
                logmsg("Solved using index %s with odds ratio %g\n",
                       solver->best_index->indexname,
                       solver->best_match.logodds);
                // WCS is solver->best_match.wcstan
                wcs = &(solver->best_match.wcstan);
                // center
                tan_pixelxy2radec(wcs, imagecx, imagecy, &ra, &dec);
                pscale = tan_pixel_scale(wcs);
                logmsg("Image center is RA,Dec = (%g,%g) degrees, size is %.2g x %.2g arcmin.\n",
                       ra, dec, arcsec2arcmin(pscale * imagew), arcsec2arcmin(pscale * imageh));

                // Print the matched stars...
                int j;
                for (j=0; j<match->nfield; j++) {
                    double fx,fy,fra,fdec;
                    double rx,ry,rra,rdec;
                    double weight;
                    int ti, ri;
                    ri = match->theta[j];
                    if (ri < 0)
                        continue;
                    ti = j;
                    rra  = match->refradec[2*ri+0];
                    rdec = match->refradec[2*ri+1];
                    if (!tan_radec2pixelxy(wcs, rra, rdec, &rx, &ry))
                        continue;
                    fx = match->fieldxy[2*ti+0];
                    fy = match->fieldxy[2*ti+1];
                    tan_pixelxy2radec(wcs, fx, fy, &fra, &fdec);
                    weight = verify_logodds_to_weight(match->matchodds[j]);
                    logmsg("Match: field xy %.1f,%.1f, radec %.2f,%.2f; index xy %.1f,%.1f, radec %.2f,%.2f, weight %.3f\n",
                           fx, fy, fra, fdec, rx, ry, rra, rdec, weight);
                }

                // Free the items allocated in record_match_callback.
                blind_free_matchobj(match);

            } else {
                logmsg("Failed to solve.\n");
            }
        }

    skip:
        solver_cleanup_field(solver);
        solver_clear_indexes(solver);

        il_free(hplist);

        t1 = timenow();
        logmsg("That took %g seconds\n", t1-t0);
        logmsg("Sleeping...\n");
        sleep(1);
        logmsg("Starting!\n");
    }

    sl_free2(imagefiles);
    engine_free(engine);
    solver_free(solver);
    return 0;
}

