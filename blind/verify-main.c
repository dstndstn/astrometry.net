/*
 This file is part of the Astrometry.net suite.
 Copyright 2009 Dustin Lang, David W. Hogg.

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
 Runs the verification procedure in stand-alone mode.
 */

#include "matchfile.h"
#include "matchobj.h"
#include "index.h"
#include "verify2.h"
#include "xylist.h"
#include "log.h"
#include "errors.h"
#include "mathutil.h"

static const char* OPTIONS = "hvi:m:f:";

static void print_help(const char* progname) {
	printf("Usage:   %s\n"
	       "   -m <match-file>\n"
		   "   -i <index-file>\n"
		   "   -f <xylist-file>\n"
           "   [-v]: verbose\n"
	       "\n", progname);
}

int main(int argc, char** args) {
	int argchar;
	int loglvl = LOG_MSG;
	char* indexfn = NULL;
	char* matchfn = NULL;
	char* xyfn = NULL;

	index_t* index;
	matchfile* mf;
	MatchObj* mo;
	verify_field_t* vf;
	starxy_t* fieldxy;
	xylist_t* xyls;

	double pix2 = 1.0;
	double distractors = 0.25;
	double fieldW=0, fieldH=0;
	double logbail = log(-1e100);
	double logkeep = log(1e12);
	bool growvariance = TRUE;
	bool fake = FALSE;

	while ((argchar = getopt(argc, args, OPTIONS)) != -1)
		switch (argchar) {
		case 'f':
			xyfn = optarg;
			break;
        case 'i':
			indexfn = optarg;
			break;
		case 'm':
			matchfn = optarg;
			break;
		case 'h':
			print_help(args[0]);
			exit(0);
		case 'v':
			loglvl++;
			break;
		}

	log_init(loglvl);

	if (!indexfn || !matchfn || !xyfn) {
		logerr("You must specify index (-i) and matchfile (-m) and field xylist (-f).\n");
		print_help(args[0]);
		exit(-1);
	}

	mf = matchfile_open(matchfn);
	if (!mf) {
		ERROR("Failed to read match file %s", matchfn);
		exit(-1);
	}

	index = index_load(indexfn, 0);
	if (!index) {
		ERROR("Failed to open index %s", indexfn);
		exit(-1);
	}

	xyls = xylist_open(xyfn);
	if (!xyls) {
		ERROR("Failed to open xylist %s", xyfn);
		exit(-1);
	}
	fieldW = xylist_get_imagew(xyls);
	fieldH = xylist_get_imageh(xyls);

    logmsg("Field W,H = %g, %g\n", fieldW, fieldH);

	mo = matchfile_read_match(mf);
	if (!mo) {
		ERROR("Failed to read object from match file.");
		exit(-1);
	}
    mo->wcstan.imagew = fieldW;
    mo->wcstan.imageh = fieldH;

	fieldxy = xylist_read_field(xyls, NULL);
	if (!fieldxy) {
		ERROR("Failed to read a field from xylist %s", xyfn);
		exit(-1);
	}

	vf = verify_field_preprocess(fieldxy);

    pix2 += square(index->meta.index_jitter / mo->scale);

    mo->logodds = 0.0;
	verify_hit(index, mo, NULL, vf,
			   pix2, distractors, fieldW, fieldH,
			   logbail, growvariance,
			   index_get_quad_dim(index), fake,
			   logkeep);
    logmsg("Logodds: %g\n", mo->logodds);
    logmsg("Odds: %g\n", exp(mo->logodds));

	verify_field_free(vf);

	starxy_free(fieldxy);

	xylist_close(xyls);
	index_close(index);
	matchfile_close(mf);

	return 0;
}
