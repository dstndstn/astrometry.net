/*
 This file is part of the Astrometry.net suite.
 Copyright 2007, 2008 Dustin Lang, Keir Mierle and Sam Roweis.

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
 * Accepts an augmented xylist that describes a field or set of fields to solve.
 * Reads a config file to find local indices, and merges information about the
 * indices with the job description to create an input file for 'blind'.  Runs blind
 * and merges the results.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <time.h>
#include <libgen.h>
#include <getopt.h>
#include <dirent.h>
#include <assert.h>
#include <glob.h>

#include "tic.h"
#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "blindutils.h"
#include "blind.h"
#include "log.h"
#include "qfits.h"
#include "errors.h"
#include "backend.h"
#include "an-opts.h"

static an_option_t myopts[] = {
	{'h', "help", no_argument, NULL, "print this help"},
	{'v', "verbose", no_argument, NULL, "+verbose"},
	{'c', "config",  required_argument, "file",
	 "Use this config file (default: \"backend.cfg\" in the directory ../etc/ relative to the directory containing the \"backend\" executable)"},
	{'d', "base-dir",  required_argument, "dir", 
	 "set base directory of all output filenames."},
	{'C', "cancel",  required_argument, "file", 
	 "quit solving if this file appears" },
	{'s', "solved",  required_argument, "file",
	 "write to this file when a field is solved"},
	{'E', "to-stderr", no_argument, NULL,
	 "send log message to stderr"},
	{'f', "inputs-from", required_argument, "file",
	 "read input filenames from the given file, \"-\" for stdin"},
	{'i', "index", required_argument, "file(s)",
	 "use the given index files (in addition to any specified in the config file); put in quotes to use wildcards, eg: \" -i 'index-*.fits' \""},
};

static void print_help(const char* progname, bl* opts) {
	printf("Usage:   %s [options] <augmented xylist (axy) file(s)>\n", progname);
	opts_print_help(opts, stdout, NULL, NULL);
}

int main(int argc, char** args) {
    char* default_configfn = "backend.cfg";
    char* default_config_path = "../etc";

	int c;
	char* configfn = NULL;
	int i;
	backend_t* backend;
    char* mydir = NULL;
    char* basedir = NULL;
    char* me;
    bool help = FALSE;
    sl* strings = sl_new(4);
    char* cancelfn = NULL;
    char* solvedfn = NULL;
    int loglvl = LOG_MSG;
    bool tostderr = FALSE;
    bool verbose = FALSE;
    char* infn = NULL;
    FILE* fin = NULL;
    bool fromstdin = FALSE;

	bl* opts = opts_from_array(myopts, sizeof(myopts)/sizeof(an_option_t), NULL);
	sl* inds = sl_new(4);

	while (1) {
		c = opts_getopt(opts, argc, args);
		if (c == -1)
			break;
		switch (c) {
		case 'i':
			//printf("Index %s\n", optarg);
			sl_append(inds, optarg);
			break;
		case 'd':
		  basedir = optarg;
		  break;
        case 'f':
            infn = optarg;
            fromstdin = streq(infn, "-");
            break;
        case 'E':
            tostderr = TRUE;
            break;
		case 'h':
            help = TRUE;
			break;
        case 'v':
            loglvl++;
            verbose = TRUE;
            break;
		case 's':
		  solvedfn = optarg;
        case 'C':
            cancelfn = optarg;
            break;
		case 'c':
			configfn = strdup(optarg);
			break;
		case '?':
			break;
		default:
            printf("Unknown flag %c\n", c);
			exit( -1);
		}
	}

	if (optind == argc && !infn) {
		// Need extra args: filename
		printf("You must specify at least one input file!\n\n");
		help = TRUE;
	}
	if (help) {
		print_help(args[0], opts);
		exit(0);
	}

    log_init(loglvl);
    if (tostderr)
        log_to(stderr);

    if (infn) {
        logverb("Reading input filenames from %s\n", (fromstdin ? "stdin" : infn));
        if (!fromstdin) {
            fin = fopen(infn, "rb");
            if (!fin) {
                ERROR("Failed to open file %s for reading input filenames", infn);
                exit(-1);
            }
        } else
            fin = stdin;
    }

	backend = backend_new();

    // directory containing the 'backend' executable:
    me = find_executable(args[0], NULL);
    if (!me)
        me = strdup(args[0]);
    mydir = sl_append(strings, dirname(me));
    free(me);

	// Read config file
    if (!configfn) {
        int i;
        sl* trycf = sl_new(4);
        sl_appendf(trycf, "%s/%s/%s", mydir, default_config_path, default_configfn);
        sl_appendf(trycf, "%s/%s", mydir, default_configfn);
        sl_appendf(trycf, "./%s", default_configfn);
        sl_appendf(trycf, "./%s/%s", default_config_path, default_configfn);
        for (i=0; i<sl_size(trycf); i++) {
            char* cf = sl_get(trycf, i);
            if (file_exists(cf)) {
                configfn = strdup(cf);
                logverb("Using config file \"%s\"\n", cf);
                break;
            } else {
                logverb("Config file \"%s\" doesn't exist.\n", cf);
            }
        }
        if (!configfn) {
            char* cflist = sl_join(trycf, "\n  ");
            logerr("Couldn't find config file: tried:\n  %s\n", cflist);
            free(cflist);
        }
        sl_free2(trycf);
    }

	if (backend_parse_config_file(backend, configfn)) {
        logerr("Failed to parse (or encountered an error while interpreting) config file \"%s\"\n", configfn);
		exit( -1);
	}

	if (sl_size(inds)) {
		// Expand globs.
		//sl* inds2 = sl_new(4);
		for (i=0; i<sl_size(inds); i++) {
			char* s = sl_get(inds, i);
			glob_t myglob;
			int flags = GLOB_TILDE;
			// flags |= GLOB_BRACE; // {x,y,...} expansion
			if (glob(s, flags, NULL, &myglob)) {
				SYSERROR("Failed to expand wildcards in index-file path \"%s\"", s);
				exit(-1);
			}
			//sl_append_array(inds2, (const char**)myglob.gl_pathv, myglob.gl_pathc);
			for (c=0; c<myglob.gl_pathc; c++) {
				if (backend_add_index(backend, myglob.gl_pathv[c])) {
					ERROR("Failed to add index \"%s\"", myglob.gl_pathv[c]);
					exit(-1);
				}
			}
			globfree(&myglob);
		}
	}

	if (!pl_size(backend->indexes)) {
		logerr("You must list at least one index in the config file (%s)\n", configfn);
		exit(-1);
	}

	if (backend->minwidth <= 0.0 || backend->maxwidth <= 0.0) {
		logerr("\"minwidth\" and \"maxwidth\" in the config file %s must be positive!\n", configfn);
		exit(-1);
	}

    free(configfn);

    if (!il_size(backend->default_depths)) {
        parse_depth_string(backend->default_depths,
                           "10 20 30 40 50 60 70 80 90 100 "
                           "110 120 130 140 150 160 170 180 190 200");
    }

    backend->cancelfn = cancelfn;
    backend->solvedfn = solvedfn;

    i = optind;
    while (1) {
		char* jobfn;
        job_t* job;
		struct timeval tv1, tv2;

        if (infn) {
            // Read name of next input file to be read.
            logverb("\nWaiting for next input filename...\n");
            jobfn = read_string_terminated(fin, "\n\r\0", 3, FALSE);
            if (strlen(jobfn) == 0)
                break;
        } else {
            if (i == argc)
                break;
            jobfn = args[i];
            i++;
        }
        gettimeofday(&tv1, NULL);
        logmsg("Reading file \"%s\"...\n", jobfn);
        job = backend_read_job_file(backend, jobfn);
        if (!job) {
            ERROR("Failed to read job file \"%s\"", jobfn);
            exit(-1);
        }

	if (basedir) {
	  logverb("Setting job's output base directory to %s\n", basedir);
	  job_set_output_base_dir(job, basedir);
	}

		if (backend_run_job(backend, job))
			logerr("Failed to run_job()\n");

		job_free(job);
        gettimeofday(&tv2, NULL);
		logverb("Spent %g seconds on this field.\n", millis_between(&tv1, &tv2)/1000.0);
	}

	backend_free(backend);
    sl_free2(strings);
	sl_free2(inds);

    if (fin && !fromstdin)
        fclose(fin);

    return 0;
}
