/*
  This file is part of the Astrometry.net suite.
  Copyright 2006-2008 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include "boilerplate.h"
#include "fitsioutils.h"
#include "2mass-fits.h"
#include "os-features.h"
#include "errors.h"

#define OPTIONS "ho:N:H:"

static void print_help(char* progname) {
	boilerplate_help_header(stdout);
    printf("\nUsage: %s\n"
		   "   -o <output-filename-template>     (eg, an_hp%%03i.fits)\n"
		   "  [-N <healpix-nside>]  (default = 9)\n"
		   "  [-H <allowed-healpix>] [[-H <allowed-healpix] ...]\n"
		   "      only create Astrometry.net catalogs for the given list\n"
		   "      of healpix numbers: range [0, 12*Nside*Nside-1].\n"
		   "      Default is to create catalogs for any healpix that contains\n"
		   "      stars in the input files.\n"
		   "  <input-file> [<input-file> ...]\n"
		   "\n"
		   "\nThe input files should be USNO-B1.0, Tycho-2, or 2MASS files in FITS format.\n"
		   "(To generate these files, see \"usnobtofits\", \"tycho2tofits\", and \"2masstofits\".)"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

static void init_catalog(an_catalog** cats, char* outfn, int hp, int Nside, int argc, char** args) {
	char fn[256];
    qfits_header* hdr;
	sprintf(fn, outfn, hp);
	cats[hp] = an_catalog_open_for_writing(fn);
	if (!cats[hp]) {
		fprintf(stderr, "Failed to initialized FITS output file %s.\n", fn);
		exit(-1);
	}
	// header remarks...
    hdr = an_catalog_get_primary_header(cats[hp]);
	fits_header_add_int(hdr, "HEALPIX", hp, "The healpix number of this catalog.");
	fits_header_add_int(hdr, "NSIDE", Nside, "The healpix resolution.");
	boilerplate_add_fits_headers(hdr);
	qfits_header_add(hdr, "HISTORY", "Created by the program \"build-an-catalog\"", NULL, NULL);
	qfits_header_add(hdr, "HISTORY", "build-an-catalog command line:", NULL, NULL);
	fits_add_args(hdr, args, argc);
	qfits_header_add(hdr, "HISTORY", "(end of command line)", NULL, NULL);

	if (an_catalog_write_headers(cats[hp])) {
		fprintf(stderr, "Failed to write header for FITS file %s.\n", fn);
		exit(-1);
	}
}

static int32_t tycho2_id_to_int(int tyc1, int tyc2, int tyc3) {
	int32_t id = 0;
	// tyc1 and tyc2 each fit in 14 bits.
	assert((tyc1 & ~0x3fff) == 0);
	assert((tyc2 & ~0x3fff) == 0);
	// tyc3 fits in 3 bits.
	assert((tyc3 & ~0x7) == 0);
	id |= tyc1;
	id |= (tyc2 << 14);
	id |= (tyc3 << (14 + 14));
	return id;
}

int main(int argc, char** args) {
	char* outfn = NULL;
	int c;
	int Nside = 9;
	int i, HP, j;
	an_catalog** cats;
	int64_t starid;
	int version = 0;
	int nusnob = 0, ntycho = 0;
	int n2mass = 0;
	il* allowed_hps = NULL;
    sl* inputfiles = sl_new(4);

    fits_use_error_system();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case '?':
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'H':
			if (!allowed_hps)
				allowed_hps = il_new(16);
			il_append(allowed_hps, atoi(optarg));
			break;
		case 'N':
			Nside = atoi(optarg);
			break;
		case 'o':
			outfn = optarg;
			break;
		}
    }

	for (; optind<argc; optind++) {
        sl_append(inputfiles, args[optind]);
    }

	if (!outfn || !sl_size(inputfiles)) {
		print_help(args[0]);
		exit(-1);
	}

	if (Nside < 1) {
		fprintf(stderr, "Nside must be >= 1.\n");
		print_help(args[0]);
		exit(-1);
	}

	HP = 12 * Nside * Nside;

	// check that "allowed" healpixes (-H) are within range.
	if (allowed_hps) {
		for (i=0; i<il_size(allowed_hps); i++) {
            int hp = il_get(allowed_hps, i);
			if (hp >= HP) {
				fprintf(stderr, "Healpix %i is larger than 12*Nside*Nside-1 (%i).\n", hp, HP);
				exit(-1);
			}
		}
	}

    printf("Input file last-modification times:\n");
    for (j=0; j<sl_size(inputfiles); j++) {
        char* infn = sl_get(inputfiles, j);
        char utcdate[256];
        if (file_get_last_modified_string(infn, "%a, %d  %b %Y %H:%M:%S %z", TRUE, utcdate, sizeof(utcdate))) {
            ERROR("Failed to get last-modified string for input file \"%s\"", infn);
            exit(-1);
        }
        printf("%s -- %s\n", utcdate, infn);
    }

    printf("Input file FITS headers:\n");
    for (j=0; j<sl_size(inputfiles); j++) {
        char* infn = sl_get(inputfiles, j);
        qfits_header* hdr;
        printf("\n----FITS header for %s----\n", infn);
        hdr = qfits_header_read(infn);
        qfits_header_list(hdr, stdout);
        qfits_header_destroy(hdr);
        printf("--------------------------\n");
    }

	cats = calloc(HP, sizeof(an_catalog*));

	starid = 0;

    for (j=0; j<sl_size(inputfiles); j++) {
		char* infn;
		usnob_fits* usnob = NULL;
		tycho2_fits* tycho = NULL;
		twomass_fits* twomass = NULL;
		qfits_header* hdr;
		anbool is_usnob = FALSE;
		anbool is_tycho = FALSE;
		anbool is_2mass = FALSE;
		an_entry an;
		int hp;

		infn = sl_get(inputfiles, j);
		printf("Opening catalog file %s...\n", infn);
		hdr = qfits_header_read(infn);
		if (!hdr) {
			ERROR("Couldn't read FITS header from file %s.\n", infn);
			exit(-1);
		}
		is_usnob = qfits_header_getboolean(hdr, "USNOB", 0);
		if (!is_usnob) {
			is_tycho = qfits_header_getboolean(hdr, "TYCHO_2", 0);
			if (!is_tycho) {
				is_2mass = qfits_header_getboolean(hdr, "2MASS", 0);
			}
		}
		qfits_header_destroy(hdr);
		if (!is_usnob && !is_tycho && !is_2mass) {
			// guess...
			printf("Guessing catalog type (this may generate a warning)...\n");
			usnob = usnob_fits_open(infn);
			if (!usnob) {
				tycho = tycho2_fits_open(infn);
				if (!tycho) {
					twomass = twomass_fits_open(infn);
					if (!twomass) {
						ERROR("Couldn't figure out what catalog file %s came from.\n", infn);
						exit(-1);
					}
				}
			}
		} else if (is_usnob) {
			usnob = usnob_fits_open(infn);
			if (!usnob) {
				ERROR("Couldn't open USNO-B catalog: %s\n", infn);
				exit(-1);
			}
		} else if (is_tycho) {
			tycho = tycho2_fits_open(infn);
			if (!tycho) {
				ERROR("Couldn't open Tycho-2 catalog: %s\n", infn);
				exit(-1);
			}
		} else if (is_2mass) {
			twomass = twomass_fits_open(infn);
			if (!twomass) {
				ERROR("Couldn't open 2MASS catalog: %s\n", infn);
				exit(-1);
			}
		}

		/*
		  USNO-B and Tycho-2 are actually mutually exclusive by design
		  (of USNO-B); USNO-B cut out a patch of sky around each Tycho-2
		  star (in the main catalog & first supplement) and pasted in the
		  Tycho-2 entry.
		  Therefore, we don't need to correlate stars between the catalogs.
		*/
		if (usnob) {
			usnob_entry* entry;
			int N = usnob_fits_count_entries(usnob);
			int anSpikesFound = 0;
			int spikesFound = 0;
            int tychoStars = 0;
			printf("Reading %i entries from USNO-B catalog file %s\n", N, infn);
			for (i=0; i<N; i++) {
				int ob, j;

				if (!(i % 100000)) {
					printf(".");
					fflush(stdout);
				}
				entry = usnob_fits_read_entry(usnob);
				if (!entry) {
					ERROR("Failed to read USNO-B entry.\n");
					exit(-1);
				}
				if (!entry->ndetections) {
					// Tycho-2 star.  Ignore it.
                    tychoStars++;
					continue;
                }
				if (entry->diffraction_spike) {
					// may be a diffraction spike.  Ignore it.
					spikesFound++;
					continue;
                }
				if (entry->an_diffraction_spike) {
					anSpikesFound++;
					continue;
				}

				hp = radecdegtohealpix(entry->ra, entry->dec, Nside);
				if (allowed_hps && !il_contains(allowed_hps, hp))
					continue;

				memset(&an, 0, sizeof(an));

				an.ra  = entry->ra;
				an.dec = entry->dec;
				an.motion_ra  = entry->pm_ra;
				an.motion_dec = entry->pm_dec;
				an.sigma_ra  = entry->sigma_ra;
				an.sigma_dec = entry->sigma_dec;
				an.sigma_motion_ra  = entry->sigma_pm_ra;
				an.sigma_motion_dec = entry->sigma_pm_dec;

				an.id = an_catalog_get_id(version, starid);
				starid++;

				ob = 0;
				for (j=0; j<5; j++) {
					if (entry->obs[j].field == 0)
						continue;
					an.obs[ob].mag = entry->obs[j].mag;
					// estimate from USNO-B paper section 5: photometric calibn
					an.obs[ob].sigma_mag = 0.25;
					an.obs[ob].id = entry->usnob_id;
					an.obs[ob].catalog = AN_SOURCE_USNOB;
					an.obs[ob].band = usnob_get_survey_band(entry->obs[j].survey);
					ob++;
				}
				an.nobs = ob;

				if (!cats[hp])
					// write the header for this healpix's catalog...
					init_catalog(cats, outfn, hp, Nside, argc, args);

				an_catalog_write_entry(cats[hp], &an);
				nusnob++;
			}
			usnob_fits_close(usnob);
			printf("\n");
			printf("Tycho-2 stars ignored: %d\n", tychoStars);
			printf("USNOB diffraction spikes ignored: %d\n", spikesFound);
			printf("Astrometry.net diffraction spikes ignored: %d\n", anSpikesFound);

		} else if (tycho) {
			tycho2_entry* entry;
			int N = tycho2_fits_count_entries(tycho);
			printf("Reading %i entries from Tycho-2 catalog file %s\n", N, infn);
			for (i=0; i<N; i++) {
				int ob;
				if (!(i % 100000)) {
					printf(".");
					fflush(stdout);
				}
				entry = tycho2_fits_read_entry(tycho);
				if (!entry) {
					ERROR("Failed to read Tycho-2 entry.\n");
					exit(-1);
				}

				hp = radecdegtohealpix(entry->ra, entry->dec, Nside);
				if (allowed_hps && !il_contains(allowed_hps, hp))
					continue;

				memset(&an, 0, sizeof(an));

				an.ra  = entry->ra;
				an.dec = entry->dec;
				an.sigma_ra  = entry->sigma_ra;
				an.sigma_dec = entry->sigma_dec;
				an.motion_ra  = entry->pm_ra;
				an.motion_dec = entry->pm_dec;
				an.sigma_motion_ra  = entry->sigma_pm_ra;
				an.sigma_motion_dec = entry->sigma_pm_dec;

				an.id = an_catalog_get_id(version, starid);
				starid++;

				ob = 0;
				if (entry->mag_BT != 0.0) {
					an.obs[ob].catalog = AN_SOURCE_TYCHO2;
					an.obs[ob].band = 'B';
					an.obs[ob].id = tycho2_id_to_int(entry->tyc1, entry->tyc2, entry->tyc3);
					an.obs[ob].mag = entry->mag_BT;
					an.obs[ob].sigma_mag = entry->sigma_BT;
					ob++;
				}
				if (entry->mag_VT != 0.0) {
					an.obs[ob].catalog = AN_SOURCE_TYCHO2;
					an.obs[ob].band = 'V';
					an.obs[ob].id = tycho2_id_to_int(entry->tyc1, entry->tyc2, entry->tyc3);
					an.obs[ob].mag = entry->mag_VT;
					an.obs[ob].sigma_mag = entry->sigma_VT;
					ob++;
				}
				if (entry->mag_HP != 0.0) {
					an.obs[ob].catalog = AN_SOURCE_TYCHO2;
					an.obs[ob].band = 'H';
					an.obs[ob].id = tycho2_id_to_int(entry->tyc1, entry->tyc2, entry->tyc3);
					an.obs[ob].mag = entry->mag_HP;
					an.obs[ob].sigma_mag = entry->sigma_HP;
					ob++;
				}
				an.nobs = ob;
				if (!an.nobs) {
					ERROR("Tycho entry %i: no observations!\n", i);
					continue;
				}

				if (!cats[hp])
					init_catalog(cats, outfn, hp, Nside, argc, args);

				an_catalog_write_entry(cats[hp], &an);
				ntycho++;
			}
			tycho2_fits_close(tycho);
			printf("\n");

		} else if (twomass) {
			twomass_entry* entry;
			int N = twomass_fits_count_entries(twomass);
			printf("Reading %i entries from 2MASS catalog file %s\n", N, infn);
			for (i=0; i<N; i++) {
				int ob;
				if (!(i % 100000)) {
					printf(".");
					fflush(stdout);
				}
				entry = twomass_fits_read_entry(twomass);
				if (!entry) {
					ERROR("Failed to read 2MASS entry.\n");
					exit(-1);
				}

				if (entry->minor_planet)
					continue;
				
				hp = radecdegtohealpix(entry->ra, entry->dec, Nside);
				if (allowed_hps && !il_contains(allowed_hps, hp))
					continue;

				memset(&an, 0, sizeof(an));

				an.ra  = entry->ra;
				an.dec = entry->dec;
				an.sigma_ra =
					sqrt(square(cos(deg2rad(entry->err_angle)) * entry->err_major) +
						 square(sin(deg2rad(entry->err_angle)) * entry->err_minor));
				an.sigma_dec =
					sqrt(square(sin(deg2rad(entry->err_angle)) * entry->err_major) +
						 square(cos(deg2rad(entry->err_angle)) * entry->err_minor));

				an.id = an_catalog_get_id(version, starid);
				starid++;

				ob = 0;
				if ((entry->j_quality != TWOMASS_QUALITY_NO_BRIGHTNESS) &&
					(entry->j_cc == TWOMASS_CC_NONE)) {
					an.obs[ob].catalog = AN_SOURCE_2MASS;
					an.obs[ob].band = 'J';
					an.obs[ob].id = entry->key;
					an.obs[ob].mag = entry->j_m;
					an.obs[ob].sigma_mag = entry->j_msigcom;
					ob++;
				}
				if ((entry->h_quality != TWOMASS_QUALITY_NO_BRIGHTNESS) &&
					(entry->h_cc == TWOMASS_CC_NONE)) {
					an.obs[ob].catalog = AN_SOURCE_2MASS;
					an.obs[ob].band = 'H';
					an.obs[ob].id = entry->key;
					an.obs[ob].mag = entry->h_m;
					an.obs[ob].sigma_mag = entry->h_msigcom;
					ob++;
				}
				if ((entry->k_quality != TWOMASS_QUALITY_NO_BRIGHTNESS) &&
					(entry->k_cc == TWOMASS_CC_NONE)) {
					an.obs[ob].catalog = AN_SOURCE_2MASS;
					an.obs[ob].band = 'K';
					an.obs[ob].id = entry->key;
					an.obs[ob].mag = entry->k_m;
					an.obs[ob].sigma_mag = entry->k_msigcom;
					ob++;
				}
				an.nobs = ob;
				if (!an.nobs) {
					//ERROR("2MASS entry %i: no valid observations.\n", i);
					continue;
				}

				if (!cats[hp])
					init_catalog(cats, outfn, hp, Nside, argc, args);

				an_catalog_write_entry(cats[hp], &an);
				n2mass++;
			}
			twomass_fits_close(twomass);
			printf("\n");
		}

		// update and sync each output file...
		for (i=0; i<HP; i++) {
			if (!cats[i]) continue;
            an_catalog_sync(cats[i]);
		}
	}

	printf("Read %i USNO-B objects, %i Tycho-2 objects and %i 2MASS objects.\n",
		   nusnob, ntycho, n2mass);

	for (i=0; i<HP; i++) {
        //qfits_header* hdr;
		if (!cats[i]) continue;
        //hdr = an_catalog_get_primary_header(cats[i]);
		//fits_header_mod_int(hdr, "NOBJS", an_catalog_count_entries(cats[i]), "Number of objects in this catalog.");
		if (an_catalog_fix_headers(cats[i]) ||
			an_catalog_close(cats[i])) {
			ERROR("Error fixing the header or closing AN catalog for healpix %i.\n", i);
		}
	}
	free(cats);

	if (allowed_hps)
		il_free(allowed_hps);

    sl_free2(inputfiles);

	return 0;
}

