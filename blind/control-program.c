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

#include "backend.h"

static const char* OPTIONS = "hc:v";

static void print_help(const char* progname) {
	printf("Usage:   %s [options]\n"
	       "   [-c <backend config file>]  (default: \"../etc/backend.cfg\" relative to this executable)\n"
           "   [-v]: verbose\n"
	       "\n", progname);
}

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


    while (1) {
		int option_index = 0;
		c = getopt_long(argc, args, OPTIONS, long_options, &option_index);
		if (c == -1)
			break;
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
		case '?':
		default:
            printf("Unknown flag %c\n", c);
			exit( -1);
		}
	}

	if (optind != argc) {
		printf("Didn't understand extra args (starting with \"%s\")\n", args[optind]);
        print_help(args[0]);
        exit(-1);
	}
    log_init(loglvl);

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
    if (!backend->inparallel) {
        logerr("Forcing indexes_inparallel.\n");
        backend->inparallel = TRUE;
    }

    /*
     if (backend->minwidth <= 0.0 || backend->maxwidth <= 0.0) {
     logerr("\"minwidth\" and \"maxwidth\" in the config file %s must be positive!\n", configfn);
     exit( -1);
     }
     */
    /*
     if (!il_size(backend->default_depths)) {
     parse_depth_string(backend->default_depths,
     "10 20 30 40 50 60 70 80 90 100 "
     "110 120 130 140 150 160 170 180 190 200");
     }
     */

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
        solver_t* sp;
        double app_min, app_max;
        double qsf_min = 0.1;
        int i, N;
        sip_t* sip = NULL;
        double centerxyz[3];
        int centerhp;

        sp = solver_new();
        solver_set_default_values(sp);

        // compute scale range in arcseconds per pixel.
        app_min = arcmin2arcsec(arcmin_min / imagew);
        app_max = arcmin2arcsec(arcmin_max / imagew);
        sp->funits_lower = app_min;
        sp->funits_upper = app_max;

        // If you want to look at only a limited number of sources:
        // sp->endobj = 20;

        // don't try teeny-tiny quads.
        sp->quadsize_min = qsf_min * MIN(imagew, imageh);

        sp->userdata = sp;
        sp->record_match_callback = match_callback;

        // Which indexes should we use to verify the existing WCS?
        sip_pixelxy2xyzarr(sip, imagew/2.0, imageh/2.0, centerxyz);

        N = pl_size(backend->indexes);
        for (i=0; i<N; i++) {
            index_t* index = pl_get(backend->indexes);
            index_meta_t* meta = &(index->meta);
            int centerhp;

            centerhp = xyzarrtohealpix(centerxyz, meta->hpnside);


            solver_add_index(sp, index);
        }

        // sp->timer_callback = timer_callback;

        solver_preprocess_field(sp);
        solver_verify_sip_wcs(sp, sip);

        solver_run(sp);
        solver_free_field(sp);

        solver_free(sp);
    }











	for (i = optind; i < argc; i++) {
		char* jobfn;
        job_t* job;

		jobfn = args[i];
        logverb("Reading job file \"%s\"...\n", jobfn);
        job = backend_read_job_file(backend, jobfn);
        if (!job) {
            ERROR("Failed to read job file \"%s\"", jobfn);
            exit(-1);
        }

		if (backend_run_job(backend, job))
			logerr("Failed to run_job()\n");

		job_free(job);
	}

	backend_free(backend);
    return 0;
}

