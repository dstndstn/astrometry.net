/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "index.h"
#include "fitstable.h"
#include "errors.h"
#include "starutil.h"

static const char* OPTIONS = "hi:o:";

void printHelp(char* progname) {
    fprintf(stderr, "Usage: %s\n"
            "   -i <index-filename>\n"
            "   -o <output-filename>\n"
            "\n"
            "Given an index, writes FITS tables of the star (RA,Dec)s\n"
            "  and the sets of stars that compose the quads.\n"
            "\n", progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* progname = argv[0];
    index_t* index;
    fitstable_t* table;
    char* indexfn = NULL;
    char* outfn = NULL;
    int i, D, N;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1) {
        switch (argchar) {
        case 'i':
            indexfn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        default:
        case 'h':
            printHelp(progname);
            exit(0);
        }
    }

    if (!(indexfn && outfn)) {
        printHelp(progname);
        exit(-1);
    }

    index = index_load(indexfn, 0, NULL);
    if (!index) {
        ERROR("Failed to open index");
        exit(-1);
    }

    table = fitstable_open_for_writing(outfn);
    if (!table) {
        ERROR("Failed to open output file for writing");
        exit(-1);
    }
    D = index_get_quad_dim(index);
    fitstable_add_write_column_array(table, fitscolumn_i32_type(), D,
                                     "quads", "");
    if (fitstable_write_primary_header(table) ||
        fitstable_write_header(table)) {
        ERROR("Failed to write headers");
        exit(-1);
    }

    N = index_nquads(index);
    for (i=0; i<N; i++) {
        unsigned int quad[D];
        quadfile_get_stars(index->quads, i, quad);
        if (fitstable_write_row(table, quad)) {
            ERROR("Failed to write quad %i", i);
            exit(-1);
        }
    }

    if (fitstable_fix_header(table)) {
        ERROR("Failed to fix quad header");
        exit(-1);
    }

    fitstable_next_extension(table);
    fitstable_clear_table(table);

    // write star RA,Dec s.
    fitstable_add_write_column(table, fitscolumn_double_type(), "RA", "deg");
    fitstable_add_write_column(table, fitscolumn_double_type(), "Dec", "deg");

    if (fitstable_write_header(table)) {
        ERROR("Failed to write star header");
        exit(-1);
    }

    N = index_nstars(index);
    for (i=0; i<N; i++) {
        double xyz[3];
        double ra, dec;
        startree_get(index->starkd, i, xyz);
        xyzarr2radecdeg(xyz, &ra, &dec);
        if (fitstable_write_row(table, &ra, &dec)) {
            ERROR("Failed to write star %i", i);
            exit(-1);
        }
    }

    if (fitstable_fix_header(table) ||
        fitstable_close(table)) {
        ERROR("Failed to fix star header and close");
        exit(-1);
    }
    

    index_close(index);

    return 0;
}
