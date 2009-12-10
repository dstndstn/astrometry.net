/*
  This file is part of the Astrometry.net suite.
  Copyright 2009 Dustin Lang.

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

#include "build-index.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "starutil.h"

const char* OPTIONS = "hvi:o:N:l:u:S:fU:H:s:m:n:r:d:p:R:L:EI:MT";

static void print_help(char* progname) {
	boilerplate_help_header(stdout);
	printf("\nUsage: %s\n"
	       "      -i <input-FITS-catalog>  input: source RA,DEC, etc\n"
		   "      -o <output-index>        output filename for index\n"
		   "      -N <nside>            healpix Nside for quad-building\n"
		   "      -l <min-quad-size>    minimum quad size (arcminutes)\n"
		   "      -u <max-quad-size>    maximum quad size (arcminutes)\n"
		   "      [-S]: sort column (default: assume already sorted)\n"
		   "      [-f]: sort in descending order (eg, for FLUX); default ascending (eg, for MAG)\n"
		   "      [-U]: healpix Nside for uniformization (default: same as -n)\n"
		   "      [-H <big healpix>]; default is all-sky\n"
           "      [-s <big healpix Nside>]; default is 1\n"
		   "      [-m <margin>]: add a margin of <margin> healpixels; default 0\n"
		   "      [-n <sweeps>]    (ie, number of stars per fine healpix grid cell); default 10\n"
		   "      [-r <dedup-radius>]: deduplication radius in arcseconds; default no deduplication\n"
		   "\n"
		   "      [-d <dimquads>] number of stars in a \"quad\" (default 4).\n"
		   "      [-p <passes>]   number of rounds of quad-building (ie, # quads per healpix cell, default 1)\n"
		   "      [-R <reuse-times>] number of times a star can be used.\n"
		   "      [-L <max-reuses>] make extra passes through the healpixes, increasing the \"-r\" reuse\n"
		   "                     limit each time, up to \"max-reuses\".\n"
		   "      [-E]: scan through the catalog, checking which healpixes are occupied.\n"
		   "\n"
		   "      [-I <unique-id>] set the unique ID of this index\n"
		   "\n"
		   "      [-M]: in-memory (don't use temp files)\n"
		   "      [-T]: don't delete temp files\n"
		   "      [-v]: add verbosity.\n"
	       "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** argv) {
	int argchar;

	char* infn = NULL;
	char* indexfn = NULL;

	index_params_t myp;
	index_params_t* p;

	int loglvl = LOG_MSG;
	int i;

	p = &myp;
	build_index_defaults(p);

	while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
		switch (argchar) {
		case 'T':
			p->delete_tempfiles = FALSE;
			break;
		case 'E':
			p->scanoccupied = TRUE;
			break;
		case 'L':
			p->Nloosen = atoi(optarg);
			break;
		case 'R':
			p->Nreuse = atoi(optarg);
			break;
		case 'p':
			p->passes = atoi(optarg);
			break;
		case 'M':
			p->inmemory = TRUE;
			break;
		case 'U':
			p->UNside = atoi(optarg);
			break;
		case 'i':
            infn = optarg;
			break;
		case 'o':
			indexfn = optarg;
			break;
		case 'u':
			p->qhi = atof(optarg);
			break;
		case 'l':
			p->qlo = atof(optarg);
			break;
		case 'S':
			p->sortcol = optarg;
			break;
		case 'f':
			p->sortasc = FALSE;
			break;
		case 'H':
			p->bighp = atoi(optarg);
			break;
		case 's':
			p->bignside = atoi(optarg);
			break;
		case 'n':
			p->sweeps = atoi(optarg);
			break;
		case 'N':
			p->Nside = atoi(optarg);
			break;
		case 'r':
			p->dedup = atof(optarg);
			break;
		case 'm':
			p->margin = atoi(optarg);
			break;
		case 'd':
			p->dimquads = atoi(optarg);
			break;
		case 'I':
			p->indexid = atoi(optarg);
			break;
		case 'v':
			loglvl++;
			break;
		case 'h':
			print_help(argv[0]);
			exit(0);
		default:
			return -1;
		}
	
	log_init(loglvl);

	if (!infn || !indexfn) {
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

	if (!p->indexid)
		logmsg("Warning: you should set the unique-id for this index (with -I).\n");

	if (p->dimquads > DQMAX) {
		ERROR("Quad dimension %i exceeds compiled-in max %i.\n", p->dimquads, DQMAX);
		exit(-1);
	}

	if (build_index_files(infn, indexfn, p)) {
		exit(-1);
	}
	return 0;
}

