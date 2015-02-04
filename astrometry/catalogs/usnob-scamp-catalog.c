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

#include <unistd.h>
#include <stdio.h>
#include <math.h>

#include "scamp-catalog.h"
#include "usnob-fits.h"
//#include "healpix-utils.h"
#include "healpix.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"

const char* OPTIONS = "hu:o:A:D:r:n:RBNv";

void print_help(char* progname) {
    printf("Usage: %s\n"
           "  -u <usnob-fits-filename-pattern>: eg /path/to/usnob/usnob_hp%%03i.fits\n"
           "  -n <usnob-fits-nside>           : Nside of USNOB healpixelization.\n"
           "  -o <scamp-reference-catalog>    : output filename\n"
           "  -A <RA center>\n"
           "  -D <Dec center>\n"
           "  -r <radius in arcmin>\n"
           "  [-R]   : use Red mags\n"
           "  [-B]   : use Blue mags\n"
           "  [-N]   : use Infrared mags\n"
           "  [-v]: verbose\n"
           "\n", progname);
}


int main(int argc, char** args) {
	int c;
    char* usnobpath = NULL;
    char* scampref = NULL;
    double ra = 0.0;
    double dec = 0.0;
    double radius = 0.0;
    double xyz[3];
    double range;
    int healpixes[9];
    int nhp;
    int nside;
    int i;
    scamp_cat_t* scamp;
    anbool red, blue, infrared;
    int loglvl = LOG_MSG;

    red = blue = infrared = FALSE;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
			print_help(args[0]);
			exit(0);
        case 'u':
            usnobpath = optarg;
            break;
        case 'n':
            nside = atoi(optarg);
            break;
        case 'o':
            scampref = optarg;
            break;
        case 'A':
            ra = atora(optarg);
            break;
        case 'D':
            dec = atodec(optarg);
            break;
        case 'r':
            radius = atof(optarg);
            break;
        case 'R':
            red = TRUE;
            break;
        case 'B':
            blue = TRUE;
            break;
        case 'N':
            infrared = TRUE;
            break;
        case 'v':
            loglvl++;
            break;
        }
    }
    log_init(loglvl);

    if (ra == HUGE_VAL || dec == HUGE_VAL || !usnobpath || !scampref || radius == 0.0 || !nside) {
        print_help(args[0]);
        printf("\n\nNeed RA, Dec, USNOB path, Nside, Scamp output file, and radius.\n");
        exit(-1);
    }

    if ((red ? 1:0) + (blue ? 1:0) + (infrared ? 1:0) != 1) {
        print_help(args[0]);
        printf("Must select exactly one of Red, Blue and Infrared (-R, -B, -N)\n");
        exit(-1);
    }

    logverb("(RA,Dec) center (%g, %g) degrees\n", ra, dec);
    logverb("Search radius %g arcmin\n", radius);

    scamp = scamp_catalog_open_for_writing(scampref, TRUE);
    if (!scamp ||
        scamp_catalog_write_field_header(scamp, NULL)) {
        ERROR("Failed to open SCAMP reference catalog for writing: \"%s\"", scampref);
        exit(-1);
    }

    radecdeg2xyzarr(ra, dec, xyz);
    range = arcmin2dist(radius);
    nhp = healpix_get_neighbours_within_range(xyz, range, healpixes, nside);

    logverb("Found %i healpixes within range.\n", nhp);

    for (i=0; i<nhp; i++) {
        int hp = healpixes[i];
        char* path;
        usnob_fits* usnob;
        int j, N;
        int nspikes = 0;
        int nanspikes = 0;
        int ntycho = 0;
        int noutofrange = 0;
        int nnomag = 0;
        int nwritten = 0;

        asprintf(&path, usnobpath, hp);
        logmsg("Opening USNOB file \"%s\"\n", path);
        usnob = usnob_fits_open(path);
        if (!usnob) {
            ERROR("Failed to open USNOB file \"%s\"", path);
            exit(-1);
        }
        N = usnob_fits_count_entries(usnob);
        logmsg("Reading %i entries.\n", N);
        for (j=0; j<N; j++) {
            usnob_entry* entry;
            scamp_ref_t ref;
            float mag;

            entry = usnob_fits_read_entry(usnob);

            if (!usnob_is_usnob_star(entry)) {
                ntycho++;
                continue;
            }
            if (distsq_between_radecdeg(ra, dec, entry->ra, entry->dec) > range*range) {
                noutofrange++;
                continue;
            }
            if (entry->diffraction_spike) {
                nspikes++;
                continue;
            }
            if (entry->an_diffraction_spike) {
                nanspikes++;
                continue;
            }
            if ((red      && usnob_get_red_mag     (entry, &mag)) ||
                (blue     && usnob_get_blue_mag    (entry, &mag)) ||
                (infrared && usnob_get_infrared_mag(entry, &mag))) {
                nnomag++;
                continue;
            }
            ref.ra  = entry->ra;
            ref.dec = entry->dec;
            ref.err_a = entry->sigma_ra;
            ref.err_b = entry->sigma_dec;
            ref.mag = mag;
            // from USNOB docs.
            ref.err_mag = 0.25;
            scamp_catalog_write_reference(scamp, &ref);
            nwritten++;
        }
        usnob_fits_close(usnob);

        logmsg("Read a total of %i USNOB entries.\n", N);
        logmsg("Rejected %i Tycho-2 stars.\n", ntycho);
        logmsg("Rejected %i stars that were out of range.\n", noutofrange);
        logmsg("Rejected %i diffraction spikes (marked by USNOB).\n", nspikes);
        logmsg("Rejected %i diffraction spikes (marked by Astrometry.net).\n", nanspikes);
        logmsg("Rejected %i stars that were missing magnitude measurements.\n", nnomag);
        logmsg("Wrote %i stars.\n", nwritten);
    }
        
    if (scamp_catalog_close(scamp)) {
        ERROR("Failed to close SCAMP reference catalog \"%s\"", scampref);
        exit(-1);
    }

    return 0;
}

