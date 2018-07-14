/*
 # This file is part of libkd.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "kdtree.h"
#include "kdtree_fits_io.h"
#include "ioutils.h"
#include "fitsioutils.h"
#include "errors.h"
#include "anqfits.h"

void printHelp(char* progname) {
    printf("\nUsage: %s <input> <output>\n"
           "\n", progname);
}


const char* OPTIONS = "hvq";

int main(int argc, char** args) {
    int argchar;
    char* progname = args[0];
    kdtree_t* kd;
    char* infn;
    char* outfn;
    qfits_header* hdr;
    qfits_header* outhdr;
    int i, Next;
    FILE* fout;
    FILE* fin;
    anbool verbose = FALSE;
    char* err;
    anbool force_quad = FALSE;
    anqfits_t* anq = NULL;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            verbose = TRUE;
            break;
        case 'q':
            force_quad = TRUE;
            break;
        case 'h':
            printHelp(progname);
            exit(-1);
        }

    if (optind != argc - 2) {
        printHelp(progname);
        exit(-1);
    }

    infn = args[optind];
    outfn = args[optind+1];

    if (!strcmp(infn, outfn)) {
        printf("Sorry, in-place modification of files is not supported.\n");
        exit(-1);
    }

    if (!force_quad && ends_with(infn, ".quad.fits")) {
        printf("\nYou don't need to fix .quad.fits files.\n"
               "  (use the -q option to try anyway.)\n");
        exit(1);
    }

    printf("Reading kdtree from file %s ...\n", infn);

    errors_start_logging_to_string();
    kd = kdtree_fits_read(infn, NULL, &hdr);
    err = errors_stop_logging_to_string("\n  ");
    if (!kd) {
        printf("Failed to read kdtree from file %s:\n", infn);
        printf("  %s\n", err);
        free(err);
        exit(-1);
    }
    free(err);

    if (!kdtree_has_old_bb(kd)) {
        printf("Kdtree %s has the correct number of bounding boxes; it doesn't need fixing.\n", infn);
        exit(1);
    }

    if (verbose) {
        printf("Tree name: %s\n", kd->name);
        printf("Treetype: 0x%x\n", kd->treetype);
        printf("Data type:     %s\n", kdtree_kdtype_to_string(kdtree_datatype(kd)));
        printf("Tree type:     %s\n", kdtree_kdtype_to_string(kdtree_treetype(kd)));
        printf("External type: %s\n", kdtree_kdtype_to_string(kdtree_exttype(kd)));
        printf("N data points:  %i\n", kd->ndata);
        printf("Dimensions:     %i\n", kd->ndim);
        printf("Nodes:          %i\n", kd->nnodes);
        printf("Leaf nodes:     %i\n", kd->nbottom);
        printf("Non-leaf nodes: %i\n", kd->ninterior);
        printf("Tree levels:    %i\n", kd->nlevels);
        printf("LR array:     %s\n", (kd->lr     ? "yes" : "no"));
        printf("Perm array:   %s\n", (kd->perm   ? "yes" : "no"));
        printf("Bounding box: %s\n", (kd->bb.any ? "yes" : "no"));
        printf("Split plane:  %s\n", (kd->split.any ? "yes" : "no"));
        printf("Split dim:    %s\n", (kd->splitdim  ? "yes" : "no"));
        printf("Data:         %s\n", (kd->data.any  ? "yes" : "no"));

        if (kd->minval && kd->maxval) {
            int d;
            printf("Data ranges:\n");
            for (d=0; d<kd->ndim; d++)
                printf("  %i: [%g, %g]\n", d, kd->minval[d], kd->maxval[d]);
        }
    }

    if (verbose)
        printf("Computing bounding boxes...\n");
    kdtree_fix_bounding_boxes(kd);

    if (verbose)
        printf("Running kdtree_check...\n");
    if (kdtree_check(kd)) {
        printf("kdtree_check failed.\n");
        exit(-1);
    }

    outhdr = qfits_header_new();
    fits_append_long_comment(outhdr, "This file was processed by the fix-bb "
                             "program, part of the Astrometry.net suite.  The "
                             "extra FITS headers in the original file are "
                             "given below:");
    fits_append_long_comment(outhdr, "---------------------------------");
                          
    for (i=0; i<qfits_header_n(hdr); i++) {
        char key[FITS_LINESZ+1];
        char val[FITS_LINESZ+1];
        char com[FITS_LINESZ+1];
        qfits_header_getitem(hdr, i, key, val, com, NULL);
        if (!(fits_is_primary_header(key) ||
              fits_is_table_header(key))) {
            qfits_header_append(outhdr, key, val, com, NULL);
        }
    }
    fits_append_long_comment(outhdr, "---------------------------------");

    if (kdtree_fits_write(kd, outfn, outhdr)) {
        ERROR("Failed to write output");
        exit(-1);
    }

    if (verbose)
        printf("Finding extra extensions...\n");

    fin = fopen(infn, "rb");
    if (!fin) {
        SYSERROR("Failed to re-open input file %s for reading", infn);
        exit(-1);
    }
    fout = fopen(outfn, "ab");
    if (!fout) {
        SYSERROR("Failed to re-open output file %s for writing", outfn);
        exit(-1);
    }

    anq = anqfits_open(infn);
    if (!anq) {
        ERROR("Failed to open input file %s for reading", infn);
        exit(-1);
    }
    Next = anqfits_n_ext(anq);

    for (i=0; i<Next; i++) {
        int hoffset, hlength;
        int doffset, dlength;
        int ext = i+1;

        if (anqfits_is_table(anq, ext)) {
            qfits_table* table;
            table = anqfits_get_table(anq, ext);
            if (table &&
                (table->nc == 1) &&
                kdtree_fits_column_is_kdtree(table->col[0].tlabel))
                continue;
        }
        if (verbose)
            printf("Extension %i is not part of the kdtree.  Copying it verbatim.\n", ext);

        hoffset = anqfits_header_start(anq, i);
        hlength = anqfits_header_size (anq, i);
        doffset = anqfits_data_start(anq, i);
        dlength = anqfits_data_size (anq, i);

        if (pipe_file_offset(fin, hoffset, hlength, fout) ||
            pipe_file_offset(fin, doffset, dlength, fout)) {
            ERROR("Failed to write extension %i verbatim", ext);
            exit(-1);
        }
    }
    fclose(fin);
    if (fclose(fout)) {
        SYSERROR("Failed to close output file %s", outfn);
        exit(-1);
    }
    anqfits_close(anq);
    kdtree_fits_close(kd);
    errors_free();

    printf("Fixed file %s was written successfully.\n", outfn);

    return 0;
}
