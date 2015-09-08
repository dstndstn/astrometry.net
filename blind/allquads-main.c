/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "fitsioutils.h"
#include "starkd.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"
#include "quad-builder.h"

const char* OPTIONS = "hi:c:q:u:l:d:I:v";

static void print_help(char* progname) {
	BOILERPLATE_HELP_HEADER(stdout);
	printf("\nUsage: %s\n"
	       "      -i <input-filename>    (star kdtree (skdt.fits) input file)\n"
		   "      -c <codes-output-filename>    (codes file (code.fits) output file)\n"
           "      -q <quads-output-filename>    (quads file (quad.fits) output file)\n"
	       "     [-u <scale>]    upper bound of quad scale (arcmin)\n"
	       "     [-l <scale>]    lower bound of quad scale (arcmin)\n"
		   "     [-d <dimquads>] number of stars in a \"quad\".\n"
		   "     [-I <unique-id>] set the unique ID of this index\n\n"
		   "\nReads skdt, writes {code, quad}.\n\n"
	       , progname);
}


int main(int argc, char** argv) {
	int argchar;
	allquads_t* aq;
	int loglvl = LOG_MSG;
	int i;

	aq = allquads_init();

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'v':
			loglvl++;
			break;
		case 'd':
			aq->dimquads = atoi(optarg);
			break;
		case 'I':
			aq->id = atoi(optarg);
			break;
		case 'h':
			print_help(argv[0]);
			exit(0);
		case 'i':
            aq->skdtfn = optarg;
			break;
		case 'c':
            aq->codefn = optarg;
            break;
        case 'q':
            aq->quadfn = optarg;
            break;
		case 'u':
			aq->quad_d2_upper = arcmin2distsq(atof(optarg));
			aq->use_d2_upper = TRUE;
			break;
		case 'l':
			aq->quad_d2_lower = arcmin2distsq(atof(optarg));
			aq->use_d2_lower = TRUE;
			break;
		default:
			return -1;
		}

	log_init(loglvl);

	if (!aq->skdtfn || !aq->codefn || !aq->quadfn) {
		printf("Specify in & out filenames, bonehead!\n");
		print_help(argv[0]);
		exit( -1);
	}

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

	if (!aq->id)
		logmsg("Warning: you should set the unique-id for this index (-i).\n");

	if (aq->dimquads > DQMAX) {
		ERROR("Quad dimension %i exceeds compiled-in max %i.\n", aq->dimquads, DQMAX);
		exit(-1);
	}
	aq->dimcodes = dimquad2dimcode(aq->dimquads);

	if (allquads_open_outputs(aq)) {
		exit(-1);
	}

	if (allquads_create_quads(aq)) {
		exit(-1);
	}

	if (allquads_close(aq)) {
		exit(-1);
	}

	allquads_free(aq);

	printf("Done.\n");
	return 0;
}

