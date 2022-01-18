/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "os-features.h"
#include "index.h"
#include "starutil.h"
#include "log.h"
#include "errors.h"
#include "ioutils.h"
#include "boilerplate.h"
#include "tic.h"
#include "fitstable.h"
#include "mathutil.h"

static const char* OPTIONS = "hvr:d:R:o:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <index-files>\n"
           "    -r <ra>     (deg)\n"
           "    -d <dec>    (deg)\n"
           "    -R <radius> (deg)\n"
           "    [-o <filename>]: save results in FITS table; tag-along columns must be the same in all indices\n"
           "    [-v]: +verbose\n"
           "\n", progname);
}


int main(int argc, char **argv) {
    int argchar;
    double ra=LARGE_VAL, dec=LARGE_VAL, radius=LARGE_VAL;
    int loglvl = LOG_MSG;
    char** myargs;
    int nmyargs;
    int i;
    char* outfn = NULL;
    fitstable_t* table = NULL;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'o':
            outfn = optarg;
            break;
        case 'r':
            ra = atof(optarg);
            break;
        case 'd':
            dec = atof(optarg);
            break;
        case 'R':
            radius = atof(optarg);
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
        default:
            printHelp(argv[0]);
            return -1;
        }
    log_init(loglvl);
    nmyargs = argc - optind;
    myargs = argv + optind;

    if (nmyargs < 1) {
        printHelp(argv[0]);
        exit(-1);
    }
    if (ra == LARGE_VAL || dec == LARGE_VAL || radius == LARGE_VAL) {
        printHelp(argv[0]);
        exit(-1);
    }

    if (outfn) {
        table = fitstable_open_for_writing(outfn);
        if (!table) {
            ERROR("Failed to open output table");
            exit(-1);
        }
        if (fitstable_write_primary_header(table)) {
            ERROR("Failed to write primary header of output table");
            exit(-1);
        }
    }

    for (i=0; i<nmyargs; i++) {
        char* indexfn = myargs[i];
        index_t index;
        sl* cols;
        int* inds;
        double* radecs;
        int N;
        int j;
        fitstable_t* tagtable = NULL;

        logmsg("Reading index \"%s\"...\n", indexfn);
        if (!index_load(indexfn, 0, &index)) {
            ERROR("Failed to read index \"%s\"", indexfn);
            continue;
        }

        logmsg("Index %s: id %i, healpix %i (nside %i), %i stars, %i quads, dimquads=%i, scales %g to %g arcmin.\n",
               index.indexname, index.indexid, index.healpix, index.hpnside,
               index.nstars, index.nquads, index.dimquads,
               arcsec2arcmin(index.index_scale_lower),
               arcsec2arcmin(index.index_scale_upper));

        cols = startree_get_tagalong_column_names(index.starkd, NULL);
        {
            char* colstr = sl_join(cols, ", ");
            logmsg("Tag-along columns: %s\n", colstr);
            free(colstr);
        }

        logmsg("Searching for stars around RA,Dec (%g, %g), radius %g deg.\n",
               ra, dec, radius);
        startree_search_for_radec(index.starkd, ra, dec, radius,
                                  NULL, &radecs, &inds, &N);
        logmsg("Found %i stars\n", N);

        if (table) {
            int tagsize;
            int rowsize;
            char* rowbuf = NULL;

            if (i > 0) {
                fitstable_next_extension(table);
                fitstable_clear_table(table);
            }

            tagtable = startree_get_tagalong(index.starkd);
            if (tagtable) {
                fitstable_add_fits_columns_as_struct(tagtable);
                logverb("Input tag-along table:\n");
                if (log_get_level() >= LOG_VERB)
                    fitstable_print_columns(tagtable);
                fitstable_copy_columns(tagtable, table);
            }
            tagsize = fitstable_get_struct_size(table);
            debug("tagsize=%i\n", tagsize);
            // Add RA,Dec at the end of the row...
            fitstable_add_write_column_struct(table, fitscolumn_double_type(), 1, tagsize, fitscolumn_double_type(), "RA", "degrees");
            fitstable_add_write_column_struct(table, fitscolumn_double_type(), 1, tagsize + sizeof(double), fitscolumn_double_type(), "DEC", "degrees");
            rowsize = fitstable_get_struct_size(table);
            assert(rowsize == tagsize + 2*sizeof(double));
            debug("rowsize=%i\n", rowsize);
            rowbuf = malloc(rowsize);

            logverb("Output table:\n");
            if (log_get_level() >= LOG_VERB)
                fitstable_print_columns(table);

            if (fitstable_write_header(table)) {
                ERROR("Failed to write header of output table");
                exit(-1);
            }

            for (j=0; j<N; j++) {
                if (tagtable) {
                    if (fitstable_read_struct(tagtable, inds[j], rowbuf)) {
                        ERROR("Failed to read row %i of tag-along table", inds[j]);
                        exit(-1);
                    }
                }
                // Add RA,Dec to end of struct...
                memcpy(rowbuf + tagsize, radecs+2*j+0, sizeof(double));
                memcpy(rowbuf + tagsize + sizeof(double), radecs+2*j+1, sizeof(double));
                if (fitstable_write_struct(table, rowbuf)) {
                    ERROR("Failed to write row %i of output", j);
                    exit(-1);
                }
            }
            free(rowbuf);

            if (fitstable_fix_header(table)) {
                ERROR("Failed to fix header of output table");
                exit(-1);
            }

        }

        sl_free2(cols);
        free(radecs);
        free(inds);

        index_close(&index);
    }

    if (table) {
        if (fitstable_close(table)) {
            ERROR("Failed to close output table");
            exit(-1);
        }
    }

    return 0;
}

