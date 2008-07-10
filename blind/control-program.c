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

static const char* OPTIONS = "hvc:i:";

static void print_help(const char* progname) {
	printf("Usage:   %s [options]\n"
	       "   [-c <backend config file>]  (default: \"../etc/backend.cfg\" relative to this executable)\n"
           "   [-v]: verbose\n"
		   "    -i <FITS image filename>\n"
	       "\n", progname);
}

static char* fits_image_fn = NULL;

// If you've got a TAN or SIP WCS, set "sip".  Otherwise, be sure to set
// (racenter, deccenter).
static void get_next_field(int* nstars, double** starx, double** stary,
						   double** starflux, sip_t** sip,
						   double* ra, double* dec) {
	int i, N;
	int W, H;
	float *x, *y, *bg, *flux;
	float sigma;
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
	W = qimg.lx;
	H = qimg.ly;
    if (qimg.np != 1) {
        logmsg("Warning, image has %i planes but this program only looks at the first one.\n", qimg.np);
    }
    if (qfits_loadpix(&qimg)) {
        ERROR("Failed to read pixels from FITS image \"%s\"", fits_image_fn);
        exit(-1);
    }

	image2xy_image(NULL, qimg.fbuf, W, H, 0, 0,
				   0, 0, 0, 0, 0, 0, 0, 0,
				   &x, &y, &flux, &bg, &N, &sigma);

	*nstars = N;
	*starx = malloc(N * sizeof(double));
	*stary = malloc(N * sizeof(double));
	*starflux = malloc(N * sizeof(double));
	for (i=0; i<N; i++) {
		(*starx)[i] = x[i];
		(*stary)[i] = y[i];
		(*starflux)[i] = flux[i];
	}

	free(x);
	free(y);
	free(flux);
	free(bg);
    qfitsloader_free_buffers(&qimg);

	// Try reading SIP header...
	*sip = sip_read_header_file(fits_image_fn, NULL);
	// If that doesn't work, look for "RA" and "DEC" header cards.
	if (!*sip) {
		qfits_header* hdr = qfits_header_read(fits_image_fn);
		*ra = qfits_header_getdouble(hdr, "RA", 0.0);
		*dec = qfits_header_getdouble(hdr, "DEC", 0.0);
		qfits_header_destroy(hdr);
	}
}

#if 0
/**
 Structure passed between main loop and solver callbacks.
 struct control_t {
 backend_t* backend;
 solver_t* solver;
 };
 typedef struct control_t control_t;
 */

/**
 This callback gets called by the solver when it encounters a match
 whose log-odds ratio is above solver->logratio_record_threshold.
 */
static bool match_callback(MatchObj* mo, void* v) {
	//control_t* control = v;
	/*
	 If you want to abort:
	 control->solver->quit_now = TRUE;
	 return FALSE;
	 */
	return TRUE;
}
#endif

int main(int argc, char** args) {
    char* configfn = NULL;
    int loglvl = LOG_MSG;

    //"../etc/backend.cfg";

    // Image size in pixels.
    int imagew = 1024;
    int imageh = 1024;
    // Image angular width range, in arcminutes.
    double arcmin_width_min = 15.0;
    double arcmin_width_max = 25.0;

	backend_t* backend;

	int c;

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
		case 'i':
			fits_image_fn = optarg;
			break;
		case '?':
		default:
            printf("Unknown flag %c\n", c);
			exit( -1);
		}

	if (optind != argc) {
		printf("Didn't understand extra args (starting with \"%s\")\n", args[optind]);
        print_help(args[0]);
        exit(-1);
	}

	if (!fits_image_fn) {
		printf("Error, you must specify a FITS image file to read (with the -i argument).\n");
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
    }

	backend = backend_new();

	if (backend_parse_config_file(backend, configfn)) {
        logerr("Failed to parse (or encountered an error while interpreting) config file \"%s\"\n", configfn);
		exit( -1);
	}

	if (!pl_size(backend->indexmetas)) {
		logerr("You must list at least one index in the config file (%s)\n", configfn);
		exit( -1);
	}
    free(configfn);

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

    for (;;) {
        // Get field
        // Do source extraction
        // Get initial WCS
        // Try to verify initial WCS
        // Use initial WCS to select appropriate indexes
        // Call solver.
        solver_t* solver;
        double app_min, app_max;
        double qsf_min = 0.1;
        int i, N;
		sip_t* sip;
        double centerxyz[3];
		double racenter, deccenter;
		il* hplist = il_new(4);
		double hprange;
		starxy_t* field;

		double *starx, *stary, *starflux;
		int nstars;

		//control_t control;
		double imagecx, imagecy;

		bool solved = FALSE;

		// Get the next image...
		racenter = 0.0;
		deccenter = 0.0;
		get_next_field(&nstars, &starx, &stary, &starflux, &sip,
					   &racenter, &deccenter);

        solver = solver_new();
        solver_set_default_values(solver);

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

		// This callback gets called when a match is found...
        //solver->record_match_callback = match_callback;
		// ... if the log-odds ratio is above this value...
		// (you can set it huge; most matches are overwhelmingly good)
		solver->logratio_record_threshold = log(1e12);
		// ... and this argument is passed to the callback.
		/*
		 control.backend = backend;
		 control.solver = solver;
		 solver->userdata = &control;
		 */

		// Feed the image source coordinates to the solver...
		field = starxy_new(nstars, TRUE, FALSE);
		starxy_set_x_array(field, starx);
		starxy_set_y_array(field, stary);
		starxy_set_flux_array(field, starflux);
		starxy_sort_by_flux(field);
		solver_set_field(solver, field);

		// Where is the center of the image according to the existing WCS?
		// center of the image in pixels
		// FIXME - not quite right wrt FITS 1-indexing
		imagecx = imagew/2.0;
		imagecy = imageh/2.0;
		if (sip)
			sip_pixelxy2xyzarr(sip, imagecx, imagecy, centerxyz);
		else
			radecdeg2xyzarr(racenter, deccenter, centerxyz);

		// What is the radius of the bounding circle of a field?
		// (in units of distance on the unit sphere)
		hprange = arcsec2dist(app_max * hypot(imagew, imageh) / 2.0);

        // Which indexes should we use to verify the existing WCS?
        N = pl_size(backend->indexes);
        for (i=0; i<N; i++) {
            index_t* index = pl_get(backend->indexes, i);
            index_meta_t* meta = &(index->meta);
			int healpixes[9];
			int nhp;

			// Find nearby healpixes (at the healpix scale of this index)
			nhp = healpix_get_neighbours_within_range(centerxyz, hprange,
													  healpixes, meta->hpnside);
			il_append_array(hplist, healpixes, nhp);
			// If the index is nearby, add it.
			if (il_contains(hplist, meta->healpix)) {
				logmsg("Adding index %s\n", meta->indexname);
				solver_add_index(solver, index);
			}

			il_remove_all(hplist);
        }

        // solver->timer_callback = timer_callback;

        solver_preprocess_field(solver);

		if (sip) {
			logmsg("Trying to verify existing WCS...\n");
			solver_verify_sip_wcs(solver, sip);
			if (solver->best_match_solves) {
				// Existing WCS passed the test.
				logmsg("Existing WCS pass the verification test with odds ratio %g\n",
					   solver->best_match.logodds);
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
					   solver->best_index->meta.indexname,
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

        solver_free_field(solver);
        solver_free(solver);
		starxy_free(field);
		il_free(hplist);

        // TEMP
        break;
    }


	backend_free(backend);
    return 0;
}

