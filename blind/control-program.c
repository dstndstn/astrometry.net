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

	if (backend->minwidth <= 0.0 || backend->maxwidth <= 0.0) {
		logerr("\"minwidth\" and \"maxwidth\" in the config file %s must be positive!\n", configfn);
		exit( -1);
	}

    free(configfn);

    if (!il_size(backend->default_depths)) {
        parse_depth_string(backend->default_depths,
                           "10 20 30 40 50 60 70 80 90 100 "
                           "110 120 130 140 150 160 170 180 190 200");
    }









    for (;;) {
        // Get field
        // Do source extraction
        // Get initial WCS
        // Try to verify initial WCS
        // Use initial WCS to select appropriate indexes
        // Call solver.

        solver_t* sp;

        sp = solver_new();
        solver_set_default_values(sp);

        // sp->timer_callback = timer_callback;
        // sp->userdata = sp;
        // sp->record_match_callback = match_callback;

        //solver_compute_quad_range();
        //solver_add_index(sp, index);

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

