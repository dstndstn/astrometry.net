#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "an-catalog.h"
#include "usnob-fits.h"
#include "tycho2-fits.h"
#include "fitsioutils.h"
#include "starutil.h"
#include "healpix.h"
#include "mathutil.h"
#include "2mass-fits.h"
#include "tycho2-fits.h"
#include "errors.h"
#include "kdtree.h"
#include "log.h"

extern char *optarg;
extern int optind, opterr, optopt;

static const char* OPTIONS = "hu:2:t:";

int main(int argc, char** args) {
    int c;
    char* tychofn = NULL;
    char* usnobfn = NULL;
    char* twomassfn = NULL;

    tycho2_fits* tycho;
    usnob_fits* usnob;
    twomass_fits* twomass;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
            /*
              case '?':
              case 'h':
              print_help(args[0]);
              exit(0);
            */
            case 'u':
                usnobfn = optarg;
                break;
            case 't':
                tychofn = optarg;
                break;
            case '2':
                twomassfn = optarg;
                break;
		}
    }
    log_init(LOG_MSG);

    if (!(usnobfn && twomassfn && tychofn)) {
        logerr("Need USNOB and 2MASS and TYCHO-2 filenames.\n");
        exit(-1);
    }

    usnob = usnob_fits_open(usnobfn);
    tycho = tycho2_fits_open(tychofn);
    twomass = twomass_fits_open(twomassfn);
    if (!(usnob && tycho && twomass)) {
        ERROR("Failed to open catalogs");
        exit(-1);
    }

    logmsg("%i USNOB, %i Tycho-2, %i 2MASS\n", usnob_fits_count_entries(usnob),
           tycho2_fits_count_entries(tycho), twomass_fits_count_entries(twomass));

    return 0;
}
