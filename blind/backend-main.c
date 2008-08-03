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
#include <libgen.h>
#include <getopt.h>
#include <dirent.h>
#include <assert.h>

#include "ioutils.h"
#include "bl.h"
#include "an-bool.h"
#include "solver.h"
#include "math.h"
#include "fitsioutils.h"
#include "blindutils.h"
#include "gnu-specific.h"
#include "blind.h"
#include "log.h"
#include "qfits.h"
#include "errors.h"
#include "backend.h"

static struct option long_options[] =
    {
	    {"help",    no_argument,       0, 'h'},
        {"verbose", no_argument,       0, 'v'},
	    {"config",  required_argument, 0, 'c'},
	    {"cancel",  required_argument, 0, 'C'},
        {"to-stderr", no_argument,     0, 'E'},
        {"inputs-from", required_argument, 0, 'f'},
	    {0, 0, 0, 0}
    };

static const char* OPTIONS = "hc:i:vC:Ef:";

static void print_help(const char* progname) {
	printf("Usage:   %s [options] <augmented xylist>\n"
	       "   [-c or --config <backend config file>]  (default: \"backend.cfg\" in the directory ../etc/ relative to the directory containing the \"backend\" executable)\n"
           "   [-C or --cancel <cancel-filename>]: quit solving if the file <cancel-filename> appears.\n"
           "   [-v or --verbose]: verbose\n"
           "   [-E or --to-stderr]: send log messages to stderr\n"
           "   [-f or --inputs-from <filename>]: read input filenames from the given file, \"-\" for standard input (stdin)\n"
	       "\n", progname);
}

int main(int argc, char** args) {
    char* default_configfn = "backend.cfg";
    char* default_config_path = "../etc";

	int c;
	char* configfn = NULL;
	int i;
	backend_t* backend;
    char* mydir = NULL;
    char* me;
    bool help = FALSE;
    sl* strings = sl_new(4);
    char* cancelfn = NULL;
    int loglvl = LOG_MSG;
    bool tostderr = FALSE;
    bool verbose = FALSE;
    char* infn = NULL;
    FILE* fin = NULL;
    bool fromstdin = FALSE;

	while (1) {
		int option_index = 0;
		c = getopt_long(argc, args, OPTIONS, long_options, &option_index);
		if (c == -1)
			break;
		switch (c) {
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
		print_help(args[0]);
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

    backend->cancelfn = cancelfn;

    i = optind;
    while (1) {
		char* jobfn;
        job_t* job;

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
    sl_free2(strings);

    if (fin && !fromstdin)
        fclose(fin);

    return 0;
}
