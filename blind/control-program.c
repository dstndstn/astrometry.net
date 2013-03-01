/*
  This file is part of the Astrometry.net suite.
  Copyright 2008 Dustin Lang.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

/**

 This is a sample telescope-control program.

 */

#include <unistd.h>
#include <stdio.h>
#include <getopt.h>
#include <libgen.h>
#include <sys/param.h>
#include <math.h>

#include "backend.h"
#include "solver.h"
#include "index.h"
#include "starxy.h"
#include "matchobj.h"
#include "healpix.h"
#include "bl.h"
#include "log.h"
#include "errors.h"

// required for this sample program, maybe not for yours...
#include "qfits.h"
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
	       "   [-c <backend config file>]  (default: \"../etc/backend.cfg\" relative to this executable)\n"
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
	//int W, H;
	simplexy_t simxy;
	// For this sample program I'm just going to read from a FITS image,
	// but you probably want to grab data fresh from the CCD...
	qfitsloader qimg;
	qimg.filename = fits_image_fn;
	qimg.xtnum = 0;
	qimg.pnum = 0;
	qimg.ptype = PTYPE_FLOAT;
	qimg.map = 1;

	if (qfitsloader_init(&qimg)) {
        ERROR("Failed to read FITS image from file \"%s\"\n", fits_image_fn);
		exit(-1);
	}
	//W = qimg.lx;
	//H = qimg.ly;
    if (qimg.np != 1) {
        logmsg("Warning, image has %i planes but this program only looks at the first one.\n", qimg.np);
    }
    if (qfits_loadpix(&qimg)) {
        ERROR("Failed to read pixels from FITS image \"%s\"", fits_image_fn);
        exit(-1);
    }

	simplexy_set_defaults(&simxy);

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
    qfitsloader_free_buffers(&qimg);

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
		qfits_header* hdr = qfits_header_read(fits_image_fn);
		*ra = qfits_header_getdouble(hdr, "RA", 0.0);
		*dec = qfits_header_getdouble(hdr, "DEC", 0.0);
		qfits_header_destroy(hdr);
        logmsg("Using (RA,Dec) estimate (%g, %g)\n", *ra, *dec);
	}
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

	backend_t* backend;
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
        configfn = resolve_path("../etc/backend.cfg", mydir);
        free(mydir);
    }

	backend = backend_new();

    logmsg("Reading config file %s and loading indexes...\n", configfn);
	if (backend_parse_config_file(backend, configfn)) {
        logerr("Failed to parse (or encountered an error while interpreting) config file \"%s\"\n", configfn);
		exit( -1);
	}

	if (!pl_size(backend->indexes)) {
		logerr("You must list at least one index in the config file (%s)\n", configfn);
		exit( -1);
	}
    free(configfn);

    logmsg("Loaded %i indexes.\n", pl_size(backend->indexes));

    // For a control program you almost certainly want to be using small enough
    // indexes that they fit in memory!
	// Maybe not -- maybe most of them won't be loaded because of healpix constraints...
    if (!backend->inparallel) {
        logerr("Forcing indexes_inparallel.\n");
        backend->inparallel = TRUE;
    }

    // I assume that the backend config file only contains indexes that cover
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
        int i, N;
		sip_t* sip;
        double centerxyz[3];
		double racenter, deccenter;
		il* hplist = il_new(4);
		starxy_t* field;
		double *starx, *stary, *starflux;
		int nstars;
		double imagecx, imagecy;
		anbool solved = FALSE;
        char* img;
        double t0, t1;

		// Get the next image...
		racenter = 0.0;
		deccenter = 0.0;
        img = sl_get(imagefiles, I % sl_size(imagefiles));
        logmsg("Reading image file %s\n", img);
		get_next_field(img, &nstars, &starx, &stary, &starflux, &sip,
					   &racenter, &deccenter);

        t0 = timenow();

        solver_cleanup_field(solver);

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
			sip_pixelxy2xyzarr(sip, imagecx, imagecy, centerxyz);
        // If there's no existing WCS, assume we got an RA,Dec estimate.
		else
			radecdeg2xyzarr(racenter, deccenter, centerxyz);

        // Which indexes should we use?  Use the WCS or RA,Dec estimate to decide.
        N = pl_size(backend->indexes);
        for (i=0; i<N; i++) {
            index_t* index = pl_get(backend->indexes, i);
			if (!index_is_within_range(index, racenter, deccenter, dist2deg(hprange)))
				continue;
			logmsg("Adding index %s\n", index->indexname);
			solver_add_index(solver, index);
        }

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
			 index_t* index = pl_get(backend->indexes);
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
			} else {
				logmsg("Failed to solve.\n");
			}
		}

        solver_clear_indexes(solver);
		starxy_free(field);
		il_free(hplist);

        t1 = timenow();
        logmsg("That took %g seconds\n", t1-t0);

        logmsg("Sleeping...\n");
        sleep(1);
        logmsg("Starting!\n");
    }

    sl_free2(imagefiles);
	backend_free(backend);
    solver_free(solver);
    return 0;
}

