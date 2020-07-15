/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**

 This is a sample telescope-control program that uses Astrometry.net
 as a library.

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
#include <sys/time.h>
#include "anqfits.h"
#include "image2xy.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "tic.h"

/**

 
 control-program -c astrometry.cfg IMAGE1.fits IMAGE2.fits IMAGE3.fits

 and it loops over the images (repeatedly), reading in the pixels,
 running source detection and astrometric calibration on each one in
 turn.

 */

#define DEFAULT_IMAGEW 1024
#define DEFAULT_IMAGEH 1024
#define DEFAULT_ARCMIN_MIN 15.0
#define DEFAULT_ARCMIN_MAX 25.0

static const char* OPTIONS = "hvc:a:A:W:H:I";


static void print_help(const char* progname) {
    printf("Usage:   %s\n"
           "   [-c <astrometry config file>]  "
           "(default: \"../etc/astrometry.cfg\" relative to this executable)\n"
           "   [-a <minimum field width>] in arcminutes, default %g\n"
           "   [-A <maximum field width>] in arcminutes, default %g\n"
           "   [-W <image width> ] in pixels, default %i\n"
           "   [-H <image height>] in pixels, default %i\n"
           "   [-I]: ignore existing WCS\n"
           "   [-v]: verbose\n"
           "\n"
           "    <FITS image filename> [<FITS image filename> ...]: "
           "the images to process.\n"
           "\n"
           "\n", progname, DEFAULT_ARCMIN_MIN, DEFAULT_ARCMIN_MAX,
           DEFAULT_IMAGEW, DEFAULT_IMAGEH);
}

// This function is called by the main code to retrieve the next
// image.  In this test program, it reads image pixels from a given
// filename and runs the source detector.  In a real application you
// might already have the image data in memory, or you might have your
// own source detection algorithm.
// 
// If you've got a TAN or SIP WCS, set "sip".
// Otherwise, be sure to set *ra and *dec to the estimated image
// center and radius.  If you don't set *radec_radius, we will assume
// 0 -- that the estimated RA,Dec is within the image.  *ra, *dec, and
// *radec_radius are all in degrees.
static void get_next_field(char* fits_image_fn,
                           int* nstars, double** starx, double** stary,
                           double** starflux, sip_t** sip,
                           double* ra, double* dec, double* radec_radius);


/* Yuck: a struct required to hold data for the record_match_callback.
 Only required if you need the list of matched reference and image stars.
 */
struct callbackdata {
    solver_t* solver;
    MatchObj match;
    float max_cpu_time;
    float max_wall_time;
    double wall_start;
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

/* This callback will be called every second. */
static time_t timer_callback(void* userdata) {
    struct callbackdata* cb = userdata;
    solver_t* solver = cb->solver;
    double walltime = timenow() - cb->wall_start;
    printf("timer_callback.  CPU time used: %f, wall time %f, best logodds %f\n",
           solver->timeused, walltime, solver->best_logodds);
    if ((cb->max_wall_time > 0) && (walltime > cb->max_wall_time)) {
        printf("max wall time exceeded; exiting\n");
        solver->quit_now = TRUE;
    }
    if ((cb->max_cpu_time > 0) && (solver->timeused > cb->max_cpu_time)) {
        printf("max CPU time exceeded; exiting\n");
        solver->quit_now = TRUE;
    }
    return 1;
}

int main(int argc, char** args) {
    // Astrometry.net config file name
    char* configfn = NULL;
    // logging level
    int loglvl = LOG_MSG;
    anbool ignore = FALSE;

    // Image size in pixels.
    int imagew = DEFAULT_IMAGEW;
    int imageh = DEFAULT_IMAGEH;
    // Image angular width range, in arcminutes.
    double arcmin_width_min = DEFAULT_ARCMIN_MIN;
    double arcmin_width_max = DEFAULT_ARCMIN_MAX;

    // arcsecond-per-pixel min and max
    double app_min;
    double app_max;

    // sl = string list, list of FITS images to read
    sl* imagefiles;
    int i, I, c;

    // Astrometry engine structure that holds the index files
    engine_t* engine;
    // core Astrometry solver parameters
    solver_t* solver;

    // Parse command-line arguments
    while ((c = getopt(argc, args, OPTIONS)) != -1)
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'v':
            loglvl++;
            break;
        case 'I':
            ignore = TRUE;
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

    // Remaining command-line arguments are the FITS image filenames to read.
    imagefiles = sl_new(4);
    for (i=optind; i<argc; i++)
        sl_append(imagefiles, args[i]);

    if (!sl_size(imagefiles)) {
        printf("You must specify at least one FITS image file to read.\n");
        print_help(args[0]);
        exit(-1);
    }

    // initialize logging
    log_init(loglvl);

    // only required if you use qfits -- makes FITS errors go through
    // the normal logging channel.
    fits_use_error_system();

    // If an astrometry.net config file was not specified, assume
    // ../etc/astrometry.cfg
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

    // Initialize the Astrometry.net engine object from the config file.
    // This includes the list of index files to use.
    logmsg("Reading config file %s and loading indexes...\n", configfn);
    engine = engine_new();
    if (engine_parse_config_file(engine, configfn)) {
        logerr("Failed to parse (or encountered an error while interpreting)"
               "config file \"%s\"\n", configfn);
        exit( -1);
    }
    // Were any index files listed in the astrometry.net config file?
    if (!pl_size(engine->indexes)) {
        logerr("You must list at least one index in the config file (%s)\n",
               configfn);
        exit( -1);
    }
    free(configfn);
    logmsg("Loaded %zu indexes.\n", pl_size(engine->indexes));

    // Note: For a control program type of application, you will
    // probably want to be using small enough index files that they
    // can all fit in memory, or you want to have a good enough
    // initial guess and small enough healpixed index files that you
    // only need to load a few into memory.  Either way, for
    // performance it's very helpful to have enough memory to hold all
    // the relevant index files, and you should set the
    //     inparallel
    // option in the config file.

    // Here, we will assume that the config file only lists index
    // files that cover the range of image angular scales you are
    // interested in.

    // Furthermore, we'll assume the range of scales is small enough
    // that we can try to verify an existing WCS without worrying that
    // we'll load millions of stars.


    // Here we initialize the core astrometry solver struct, telling
    // it about the possible range of image scales.
    solver = solver_new();

    // "quad scale fraction" -- what minimum size of quadrangle
    // should be look at, as a fraction of the image size.  The
    // idea is that tiny quadrangles (in pixel space) won't
    // produce a good solution (because the positions are
    // relatively noisy).
    double qsf_min = 0.1;

    // compute scale range in arcseconds per pixel.
    app_min = arcmin2arcsec(arcmin_width_min / (double)imagew);
    app_max = arcmin2arcsec(arcmin_width_max / (double)imagew);
    // set the solver's "funits" = field (image) scale units
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

    // This determines how good a match has to be before we accept
    // it.  You can set this to be huge; most matches are
    // overwhelmingly good, assuming you have ~10 or more matched stars.
    solver_set_keep_logodds(solver, log(1e12));

    for (I=0;; I++) {
        size_t i, N;
        sip_t* sip;
        double racenter, deccenter;
        double radec_radius;

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
        cb.max_cpu_time = 0.0;
        cb.max_wall_time = 0.0;
        MatchObj* match = &(cb.match);
        memset(match, 0, sizeof(MatchObj));

        // Get the next image...
        racenter = 0.0;
        deccenter = 0.0;
        radec_radius = 0.0;
        img = sl_get(imagefiles, I % sl_size(imagefiles));
        logmsg("Reading image file %s\n", img);
        get_next_field(img, &nstars, &starx, &stary, &starflux, &sip,
                       &racenter, &deccenter, &radec_radius);

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

        // How big of a search radius should we use?  radec_radius
        // plus the maximum image size.
        radec_radius += arcsec2deg(app_max * hypot(imagew, imageh) / 2.0);

        // Which indexes should we use?  Use the WCS or RA,Dec
        // estimate to decide.
        N = pl_size(engine->indexes);
        for (i=0; i<N; i++) {
            index_t* index = pl_get(engine->indexes, i);
            if (!isinf(racenter) &&
                !index_is_within_range(index, racenter, deccenter,
                                       radec_radius))
                continue;
            logmsg("Adding index %s\n", index->indexname);
            solver_add_index(solver, index);
        }
        if (solver_n_indices(solver) == 0) {
            logmsg("No index files are within range of given RA,Dec center "
                   "and radius: (%g,%g), %g\n",
                   racenter, deccenter, radec_radius);
            goto skip;
        }

        // If you want the list of matched stars, you have to do this ugliness:
        solver->record_match_callback = record_match_callback;
        solver->userdata = &cb;

        if (sip && !ignore) {
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

        solver->timer_callback = timer_callback;
        cb.max_cpu_time = 10;
        cb.max_wall_time = 20;
        cb.wall_start = timenow();

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
                    logverb("Match: field xy %.1f,%.1f, radec %.2f,%.2f; index xy %.1f,%.1f, radec %.2f,%.2f, weight %.3f\n",
                            fx, fy, fra, fdec, rx, ry, rra, rdec, weight);
                }

                // Free the items allocated in record_match_callback.
                onefield_free_matchobj(match);

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


// If you've got a TAN or SIP WCS, set "sip".  Otherwise, be sure to set
// (racenter, deccenter).
static void get_next_field(char* fits_image_fn,
                           int* nstars, double** starx, double** stary,
                           double** starflux, sip_t** sip,
                           double* ra, double* dec, double* radec_radius) {
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
