/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#include "healpix.h"
#include "starutil.h"
#include "ioutils.h"
#include "boilerplate.h"
#include "fitsioutils.h"

#include "qfits.h"

#define OPTIONS "hn:c:e:r:d:"

void print_help(char* progname) {
	boilerplate_help_header(stdout);
    printf("usage: %s\n"
		   "  [-n <healpix-nside>]  (default: 1)\n"
           "  [-c <column-name>]    (default: HEALPIX)\n"
           "  [-e <input-extension-num>]  (default: 1)\n"
           "  [-r <RA column name>]  (default: RA)\n"
           "  [-d <DEC column name>]  (default: DEC)\n"
           "  <input-file>  <output-file>\n"
		   "\n", progname);
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int c;
	int i;
	int Nside = 1;
    char* hpcolname = "HEALPIX";
    char* infn;
    char* outfn;
    char* racolname = "RA";
    char* deccolname = "DEC";
    int racol, deccol;
    int ext = 1;
    qfits_table* table;
    qfits_table* newtable;
    qfits_header* hdr;
    qfits_header* newhdr;
    FILE* fout;
    FILE* fin;
    int start, length;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
		case '?':
        case 'h':
			print_help(args[0]);
			exit(0);
		case 'n':
			Nside = atoi(optarg);
			break;
        case 'c':
            hpcolname = optarg;
            break;
        case 'e':
            ext = atoi(optarg);
            break;
        case 'r':
            racolname = optarg;
            break;
        case 'd':
            deccolname = optarg;
            break;
        }
    }

    if (optind != argc-2) {
		print_help(args[0]);
		exit(-1);
	}
    infn  = args[optind];
    outfn = args[optind+1];

	if (Nside < 1) {
		fprintf(stderr, "Nside must be >= 1.\n");
		print_help(args[0]);
		exit(-1);
	}
    if (ext < 1) {
		fprintf(stderr, "Extension must be >= 1.\n");
		print_help(args[0]);
		exit(-1);
    }
    if (file_exists(outfn)) {
        fprintf(stderr, "Output file \"%s\" exists.  Waiting 5 seconds to give you time to hit ctrl-C...\n", outfn);
        sleep(5);
    }

    fout = fopen(outfn, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output file \"%s\": %s\n", outfn, strerror(errno));
        exit(-1);
    }
    fin = fopen(infn, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input file \"%s\": %s\n", infn, strerror(errno));
        exit(-1);
    }

    // grab the table header...
    table = qfits_table_open(infn, ext);
    if (table->tab_t != QFITS_BINTABLE) {
        fprintf(stderr, "This program only works with BINTABLEs.\n");
        exit(-1);
    }

    // check that there isn't already a "healpix" column.
    if (fits_find_column(table, hpcolname) != -1) {
        fprintf(stderr, "The table already contains a column named \"%s\".\n", hpcolname);
        exit(-1);
    }

    // find the RA,Dec columns.
    racol  = fits_find_column(table, racolname);
    deccol = fits_find_column(table, deccolname);
    if ((racol == -1) || (deccol == -1)) {
        fprintf(stderr, "RA,Dec columns \"%s\" and \"%s\" weren't found.\n", racolname, deccolname);
        exit(-1);
    }

    // ensure the RA,Dec columns are double or float.
    if (!(((table->col[racol ].atom_type == TFITS_BIN_TYPE_E) ||
           (table->col[racol ].atom_type == TFITS_BIN_TYPE_D)) &&
          ((table->col[deccol].atom_type == TFITS_BIN_TYPE_E) ||
           (table->col[deccol].atom_type == TFITS_BIN_TYPE_D)))) {
        fprintf(stderr, "RA,Dec columns must be FITS type D or E.\n");
        exit(-1);
    }

    // create a new table header with an extra column.
    newtable = qfits_table_new(outfn, table->tab_t, table->tab_w,
                               table->nc + 1, table->nr);
    // copy existing columns
    memcpy(newtable->col, table->col, sizeof(qfits_col) * table->nc);
    // add the "healpix" column.
    if (fits_add_column(newtable, table->nc, TFITS_BIN_TYPE_I, 1, "", hpcolname)) {
        fprintf(stderr, "Failed to add column.\n");
        exit(-1);
    }

    // copy the primary header exactly.
    qfits_get_hdrinfo(infn, 0, &start, &length);
    if (pipe_file_offset(fin, start, length, fout)) {
        fprintf(stderr, "Failed to copy primary header.\n");
        exit(-1);
    }

    // copy the old extension header, except override the table keywords.
    hdr = qfits_header_readext(infn, ext);
    newhdr = qfits_table_ext_header_default(newtable);
    fits_copy_non_table_headers(newhdr, hdr);
    qfits_header_dump(newhdr, fout);
    qfits_header_destroy(newhdr);
    qfits_header_destroy(hdr);

    // jump to the start of the input data...
    qfits_get_datinfo(infn, ext, &start, &length);
    if (fseeko(fin, start, SEEK_SET)) {
        fprintf(stderr, "Failed to seek to the start of the input data.\n");
        exit(-1);
    }

    for (i=0; i<table->nr; i++) {
        double ra, dec;
        int hp;
        char rowdata[table->tab_w];

        if (fread(rowdata, 1, table->tab_w, fin) != table->tab_w) {
            fprintf(stderr, "Failed to read row %i.\n", i);
            exit(-1);
        }
        if (fwrite(rowdata, 1, table->tab_w, fout) != table->tab_w) {
            fprintf(stderr, "Failed to write row %i.\n", i);
            exit(-1);
        }
        ra  = fits_get_double_val(table, racol,  rowdata);
        dec = fits_get_double_val(table, deccol, rowdata);
        hp  = radecdegtohealpix(ra, dec, Nside);
        if (fits_write_data_I(fout, hp, TRUE)) {
            fprintf(stderr, "Failed to write healpix value for row %i.\n", i);
            exit(-1);
        }
    }

    fits_pad_file(fout);
    fclose(fout);
    fclose(fin);

    qfits_table_close(table);
    qfits_table_close(newtable);

    return 0;
}
