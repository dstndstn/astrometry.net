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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>

#include "qfits.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "permutedsort.h"
#include "an-bool.h"

const char* OPTIONS = "hdf:b:";

static void printHelp(char* progname) {
    printf("Usage:   %s  <input> <output>\n"
		   "      -f <flux-column-name>  (default: FLUX) \n"
		   "      -b <background-column-name>  (default: BACKGROUND)\n"
		   "      [-d]: sort in descending order (default is ascending)\n"
           "\n", progname);
}

static int get_double_column(qfits_table* table, int col, double* result) {
    float* fdata;
    int i;
    switch (table->col[col].atom_type) {
    case TFITS_BIN_TYPE_D:
        qfits_query_column_seq_to_array(table, col, 0, table->nr,
                                        (unsigned char*)result, sizeof(double));
        return 0;
    case TFITS_BIN_TYPE_E:
        // copy it as floats, then convert to doubles.
        fdata = (float*)result;
        qfits_query_column_seq_to_array(table, col, 0, table->nr,
                                        (unsigned char*)result, sizeof(float));
        for (i=table->nr-1; i>=0; i--)
            result[i] = fdata[i];
        return 0;
    default:
        return -1;
    }
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int argchar;
	char* infn = NULL;
	char* outfn = NULL;
	char* progname = args[0];
    char* fluxcol = "FLUX";
    char* backcol = "BACKGROUND";
    bool ascending = TRUE;
	FILE* fin;
	FILE* fout;
    double *flux = NULL, *back = NULL;
    int *perm1 = NULL, *perm2 = NULL;
    bool *used = NULL;
    int Nhighwater = 0;
    int start, size, nextens, ext;

    int (*compare)(const void*, const void*);

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'f':
            fluxcol = optarg;
            break;
        case 'b':
            backcol = optarg;
            break;
        case 'd':
            ascending = FALSE;
            break;
        case '?':
        case 'h':
			printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (optind != argc-2) {
        printHelp(progname);
        exit(-1);
    }

    if (ascending)
        compare = compare_doubles;
    else
        compare = compare_doubles_desc;

    infn = args[optind];
    outfn = args[optind+1];

    fin = fopen(infn, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open input file %s: %s\n", infn, strerror(errno));
        exit(-1);
    }
    fout = fopen(outfn, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
        exit(-1);
    }

	// copy the main header exactly.
	if (qfits_get_hdrinfo(infn, 0, &start, &size)) {
		fprintf(stderr, "Couldn't get main header.\n");
		exit(-1);
	}
    assert(start == 0);
    if (pipe_file_offset(fin, 0, size, fout)) {
        fprintf(stderr, "Failed to copy main header.\n");
        exit(-1);
    }

	nextens = qfits_query_n_ext(infn);

	for (ext=1; ext<=nextens; ext++) {
		int fc, bc;
		qfits_table* table;
		qfits_col *fcol, *bcol;
		int hdrstart, hdrsize, datsize, datstart;
		int i;

		if (!qfits_is_table(infn, ext)) {
            fprintf(stderr, "Extention %i isn't a table. Skipping.\n", ext);
			continue;
		}
		table = qfits_table_open(infn, ext);
		if (!table) {
			fprintf(stderr, "failed to open table: file %s, extension %i. Skipping.\n", infn, ext);
			continue;
		}

		fc = fits_find_column(table, fluxcol);
		if (fc == -1) {
			fprintf(stderr, "Couldn't find column named \"%s\" in extension %i.  Skipping.\n", fluxcol, ext);
			continue;
		}
		fcol = table->col + fc;

		bc = fits_find_column(table, backcol);
		if (bc == -1) {
			fprintf(stderr, "Couldn't find column named \"%s\" in extension %i.  Skipping.\n", backcol, ext);
			continue;
		}
		bcol = table->col + bc;

        if (table->nr > Nhighwater) {
            free(flux);
            free(back);
            free(perm1);
            free(perm2);
            free(used);
            flux    = malloc(table->nr * sizeof(double));
            back    = malloc(table->nr * sizeof(double));
            perm1   = malloc(table->nr * sizeof(int));
            perm2   = malloc(table->nr * sizeof(int));
            used    = malloc(table->nr * sizeof(bool));
            Nhighwater = table->nr;
        }

        if (get_double_column(table, fc, flux)) {
			fprintf(stderr, "Column %s is neither FITS type D nor E.  Skipping.\n", fluxcol);
            continue;
        }
        if (get_double_column(table, bc, back)) {
			fprintf(stderr, "Column %s is neither FITS type D nor E.  Skipping.\n", backcol);
            continue;
        }

		for (i=0; i<table->nr; i++) {
			perm1[i] = i;
			perm2[i] = i;
        }

        // set back = flux + back (ie, non-background-subtracted flux)
		for (i=0; i<table->nr; i++)
            back[i] += flux[i];

        // Sort by flux...
		permuted_sort(flux, sizeof(double), compare, perm1, table->nr);

        // Sort by non-background-subtracted flux...
		permuted_sort(back, sizeof(double), compare, perm2, table->nr);

        // Copy the header as-is.
		if (qfits_get_hdrinfo(infn, ext, &hdrstart, &hdrsize) ||
			qfits_get_datinfo(infn, ext, &datstart, &datsize)) {
			fprintf(stderr, "Couldn't get extension %i header extent.\n", ext);
			exit(-1);
		}
        if (pipe_file_offset(fin, hdrstart, hdrsize, fout)) {
            fprintf(stderr, "Failed to copy the header of extension %i\n", ext);
            exit(-1);
        }

        memset(used, 0, table->nr * sizeof(bool));

        for (i=0; i<table->nr; i++) {
            int j;
            int inds[] = { perm1[i], perm2[i] };
            for (j=0; j<2; j++) {
                int index = inds[j];
                if (used[index])
                    continue;
                used[index] = TRUE;
                if (pipe_file_offset(fin, datstart + index * table->tab_w, table->tab_w, fout)) {
                    fprintf(stderr, "Failed to copy row %i.\n", index);
                    exit(-1);
                }
            }
        }

		if (fits_pad_file(fout)) {
			fprintf(stderr, "Failed to add padding to extension %i.\n", ext);
			exit(-1);
		}
	}

    free(flux);
    free(back);
    free(perm1);
    free(perm2);
    free(used);

	if (fclose(fout)) {
		fprintf(stderr, "Error closing output file: %s\n", strerror(errno));
	}
	fclose(fin);

	return 0;
}


