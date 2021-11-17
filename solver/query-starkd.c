/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdint.h>
#include <stdio.h>

#include "starkd.h"
#include "fitsioutils.h"
#include "log.h"
#include "errors.h"
#include "boilerplate.h"
#include "starutil.h"
#include "rdlist.h"

static const char* OPTIONS = "hvr:d:R:t:Io:T";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <star-kdtree-file>\n"
           "    [-o <ra-dec-list>]: write FITS table (default: print ASCII to stdout)\n"
           "    [-r <ra>] (deg)\n"
           "    [-d <dec>] (deg)\n"
           "    [-R <radius>] (deg)\n"
           "    [-t <tagalong-column>]\n"
           "    [-T]: tag-along all\n"
           "    [-I]: print indices too\n"
           "    [-v]: +verbose\n"
           "\n", progname);
}


int main(int argc, char **argv) {
    int argchar;
    startree_t* starkd;
    double ra=0.0, dec=0.0, radius=0.0;
    sl* tag = sl_new(4);
    anbool tagall = FALSE;
    char* starfn = NULL;
    int loglvl = LOG_MSG;
    char** myargs;
    int nmyargs;
    anbool getinds = FALSE;
    double* radec;
    int* inds;
    int N;
    int i;
    char* rdfn = NULL;
    pl* tagdata = pl_new(16);
    il* tagsizes = il_new(16);
    fitstable_t* tagalong = NULL;

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'o':
            rdfn = optarg;
            break;
        case 'I':
            getinds = TRUE;
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
        case 't':
            sl_append(tag, optarg);
            break;
        case 'T':
            tagall = TRUE;
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(argv[0]);
            break;
        default:
            return -1;
        }

    nmyargs = argc - optind;
    myargs = argv + optind;

    if (nmyargs != 1) {
        ERROR("Got %i arguments; expected 1.\n", nmyargs);
        printHelp(argv[0]);
        exit(-1);
    }
    starfn = myargs[0];

    log_init(loglvl);

    starkd = startree_open(starfn);
    if (!starkd) {
        ERROR("Failed to open star kdtree");
        exit(-1);
    }

    logmsg("Searching kdtree %s at RA,Dec = (%g,%g), radius %g deg.\n",
           starfn, ra, dec, radius);

    startree_search_for_radec(starkd, ra, dec, radius,
                              NULL, &radec, &inds, &N);

    logmsg("Got %i results.\n", N);

    if (!N)
        goto done;

    if (tagall) {
        int j, M;
        M = startree_get_tagalong_N_columns(starkd); 
        for (j=0; j<M; j++)
            sl_append(tag, startree_get_tagalong_column_name(starkd, j));
    }

    if (sl_size(tag)) {
        tagalong = startree_get_tagalong(starkd);
        if (!tagalong) {
            ERROR("Failed to find tag-along table in index");
            exit(-1);
        }
    }

    if (rdfn) {
        rdlist_t* rd = rdlist_open_for_writing(rdfn);
        il* colnums = il_new(16);

        if (!rd) {
            ERROR("Failed to open output file %s", rdfn);
            exit(-1);
        }
        if (rdlist_write_primary_header(rd)) {
            ERROR("Failed to write header to output file %s", rdfn);
            exit(-1);
        }

        for (i=0; i<sl_size(tag); i++) {
            const char* col = sl_get(tag, i);
            char* units;
            tfits_type type;
            int arraysize;
            void* data;
            int colnum;
            int itemsize;

            if (fitstable_find_fits_column(tagalong, col, &units, &type, &arraysize)) {
                ERROR("Failed to find column \"%s\" in index", col);
                exit(-1);
            }
            itemsize = fits_get_atom_size(type) * arraysize;
            data = fitstable_read_column_array_inds(tagalong, col, type, inds, N, NULL);
            if (!data) {
                ERROR("Failed to read data for column \"%s\" in index", col);
                exit(-1);
            }
            colnum = rdlist_add_tagalong_column(rd, type, arraysize, type, col, NULL);

            il_append(colnums, colnum);
            il_append(tagsizes, itemsize);
            pl_append(tagdata, data);
        }
        if (rdlist_write_header(rd)) {
            ERROR("Failed to write header to output file %s", rdfn);
            exit(-1);
        }

        for (i=0; i<N; i++) {
            if (rdlist_write_one_radec(rd, radec[i*2+0], radec[i*2+1])) {
                ERROR("Failed to write RA,Dec to output file %s", rdfn);
                exit(-1);
            }
        }
        for (i=0; i<sl_size(tag); i++) {
            int col = il_get(colnums, i);
            void* data = pl_get(tagdata, i);
            int itemsize = il_get(tagsizes, i);

            if (rdlist_write_tagalong_column(rd, col, 0, N, data, itemsize)) {
                ERROR("Failed to write tag-along data column %s", sl_get(tag, i));
                exit(-1);
            }
        }
        if (rdlist_fix_header(rd) ||
            rdlist_fix_primary_header(rd) ||
            rdlist_close(rd)) {
            ERROR("Failed to close output file %s", rdfn);
            exit(-1);
        }
        il_free(colnums);

    } else {
        // Header
        printf("# RA, Dec");
        if (getinds)
            printf(", index");
        for (i=0; i<sl_size(tag); i++)
            printf(", %s", sl_get(tag, i));
        printf("\n");

        for (i=0; i<sl_size(tag); i++) {
            const char* col = sl_get(tag, i);
            char* units;
            tfits_type type;
            int arraysize;
            void* data;
            int itemsize;

            if (fitstable_find_fits_column(tagalong, col, &units, &type, &arraysize)) {
                ERROR("Failed to find column \"%s\" in index", col);
                exit(-1);
            }
            itemsize = fits_get_atom_size(type) * arraysize;
            // Type could be anything. Let's convert to double for display purposes.
            data = fitstable_read_column_array_inds(tagalong, col, fitscolumn_double_type(), inds, N, NULL);
            if (!data) {
                ERROR("Failed to read data for column \"%s\" in index", col);
                exit(-1);
            }
            il_append(tagsizes, itemsize);
            pl_append(tagdata, data);
        }

        for (i=0; i<N; i++) {
            int j;
            printf("%g, %g", radec[i*2+0], radec[i*2+1]);
            if (getinds)
                printf(", %i", inds[i]);

            for (j=0; j<pl_size(tagdata); j++) {
                // We converted to 'double' above.
                double* data = pl_get(tagdata, j);
                printf(", %g", data[i]);
            }

            printf("\n");
        }
    }

 done:
    free(radec);
    free(inds);
    for (i=0; i<pl_size(tagdata); i++)
        free(pl_get(tagdata, i));
    pl_free(tagdata);
    il_free(tagsizes);

    return 0;
}


