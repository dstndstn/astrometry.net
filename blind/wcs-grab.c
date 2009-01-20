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

#include <unistd.h>

#include "log.h"
#include "errors.h"
#include "sip.h"
#include "sip_qfits.h"
#include "fitsioutils.h"
#include "boilerplate.h"
#include "qfits.h"

const char* OPTIONS = "hv";

static void print_help(char* progname) {
    boilerplate_help_header(stdout);
    printf("\nUsage:  %s <input-filename> <extension> <output-filename>\n"
           "\n\n", progname);
}

int main(int argc, char** args) {
	char* outfn = NULL;
	char* infn = NULL;
    int ext;
	int c;
    int loglvl = LOG_MSG;
    qfits_header* hdr;
    sip_t sip;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case '?':
        case 'h':
			print_help(args[0]);
			exit(0);
        case 'v':
            loglvl++;
            break;
        }
    }

    log_init(loglvl);

    if (optind != argc-3) {
        print_help(args[0]);
        exit(-1);
    }

    infn = args[optind+0];
    ext = atoi(args[optind+1]);
    outfn = args[optind+2];

    logmsg("Reading extension %i from file \"%s\"\n", ext, infn);
    hdr = qfits_header_readext(infn, ext);
    if (!hdr) {
        ERROR("Failed to read header from extension %i of file \"%s\"\n", ext, infn);
        exit(-1);
    }

    if (!sip_read_header(hdr, &sip)) {
        ERROR("Failed to read SIP header.\n");
        exit(-1);
    }

    if (sip.a_order > 0) {
        logmsg("Got SIP header.\n");
    } else {
        logmsg("Got TAN header.\n");
    }

    logmsg("Writing to file \"%s\"\n", outfn);
    if (sip_write_to_file(&sip, outfn)) {
        ERROR("Failed to write SIP header to file \"%s\".\n", outfn);
        exit(-1);
    }

    qfits_header_destroy(hdr);

    return 0;
}
