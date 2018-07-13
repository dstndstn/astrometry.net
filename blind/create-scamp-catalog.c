/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stddef.h>

#include "scamp-catalog.h"
#include "usnob-fits.h"
#include "healpix.h"
#include "starutil.h"
#include "errors.h"
#include "log.h"

const char* OPTIONS = "hv";

void print_help(char* progname) {
    printf("Usage: %s <input-FITS-table> <output-scamp-catalog>\n"
           "  Input table must contain columns:\n"
           "      RA\n"
           "      RA_ERR\n"
           "      DEC\n"
           "      DEC_ERR\n"
           "      MAG\n"
           "      MAG_ERR\n"
           "  [-v]: verbose\n"
           "\n", progname);
}


int main(int argc, char** args) {
    int c;
    char* infn = NULL;
    char* outfn = NULL;
    scamp_cat_t* scamp;
    int loglvl = LOG_MSG;
    fitstable_t* table;
    int i, N;
    tfits_type dubl = fitscolumn_double_type();
    tfits_type any  = fitscolumn_any_type();

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'v':
            loglvl++;
            break;
        }
    }
    log_init(loglvl);

    if (optind != argc - 2) {
        print_help(args[0]);
        exit(-1);
    }

    infn = args[optind];
    outfn = args[optind+1];

    logverb("Reading input from \"%s\"\n", infn);
    logverb("Will write output to \"%s\"\n", outfn);

    table = fitstable_open(infn);
    if (!table) {
        ERROR("Failed to open input file \"%s\"", infn);
        exit(-1);
    }
    N = fitstable_nrows(table);
    logverb("Input table has %i rows\n", N);

    fitstable_add_read_column_struct(table, dubl, 1, offsetof(scamp_ref_t, ra),
                                     any, "RA", TRUE);
    fitstable_add_read_column_struct(table, dubl, 1, offsetof(scamp_ref_t, dec),
                                     any, "DEC", TRUE);
    fitstable_add_read_column_struct(table, dubl, 1, offsetof(scamp_ref_t, err_a),
                                     any, "RA_ERR", TRUE);
    fitstable_add_read_column_struct(table, dubl, 1, offsetof(scamp_ref_t, err_b),
                                     any, "DEC_ERR", TRUE);
    fitstable_add_read_column_struct(table, dubl, 1, offsetof(scamp_ref_t, mag),
                                     any, "MAG", TRUE);
    fitstable_add_read_column_struct(table, dubl, 1, offsetof(scamp_ref_t, err_mag),
                                     any, "MAG_ERR", FALSE);

    fitstable_use_buffered_reading(table, sizeof(scamp_ref_t), 1000);

    if (fitstable_read_extension(table, 1)) {
        ERROR("Failed to open table from extension 1 of \"%s\"", infn);
        fitstable_error_report_missing(table);
        logmsg("Table has columns:\n");
        fitstable_print_columns(table);
        exit(-1);
    }

    scamp = scamp_catalog_open_for_writing(outfn, TRUE);
    if (!scamp ||
        scamp_catalog_write_field_header(scamp, NULL)) {
        ERROR("Failed to open SCAMP reference catalog for writing: \"%s\"", outfn);
        exit(-1);
    }

    for (i=0; i<N; i++) {
        /*
         scamp_ref_t ref;
         memset(&ref, 0, sizeof(scamp_ref_t));
         if (fitstable_read_struct(table, i, &ref)) {
         ERROR("Failed to read entry %i from input table\n", i);
         exit(-1);
         }
         */
        scamp_ref_t* ref;
        ref = fitstable_next_struct(table);
        if (!ref) {
            ERROR("Failed to read entry %i from input table\n", i);
            exit(-1);
        }

        if (scamp_catalog_write_reference(scamp, ref)) {
            ERROR("Failed to write entry %i to SCAMP catalog.\n", i);
            exit(-1);
        }
    }

    if (scamp_catalog_close(scamp)) {
        ERROR("Failed to close SCAMP reference catalog \"%s\"", outfn);
        exit(-1);
    }

    fitstable_close(table);

    return 0;
}

