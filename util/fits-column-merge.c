/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "os-features.h"
#include "anqfits.h"
#include "fitsioutils.h"
#include "boilerplate.h"
#include "ioutils.h"

char* OPTIONS = "hs:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\n\n%s  [options] <input-file-1> <input-file-2> <output-file>\n"
           "Options:\n"
           "   [-s]: suffix to add for duplicate column names in table 2\n"
           "\n",
           progname);
}


int main(int argc, char** args) {
    int argchar;
    char* afn;
    char* bfn;
    char* outfn;
    char* suffix = NULL;
    qfits_table* atable;
    qfits_table* btable;
    qfits_table* outtable;
    int i,j;
    FILE* afid;
    FILE* bfid;
    FILE* outfid;
    int aoff;
    int boff;
    char* buffer;
    qfits_header* hdr;
    qfits_header* tablehdr;
    anqfits_t* anqa = NULL;
    anqfits_t* anqb = NULL;

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 's':
            suffix = optarg;
            break;
        case 'h':
            printHelp(args[0]);
            exit(0);
        case '?':
            exit(-1);
        }

    if (optind != (argc - 3)) {
        printHelp(args[0]);
        printf("Need 3 arguments.\n");
        exit(-1);
    }

    afn = args[optind];
    bfn = args[optind+1];
    outfn = args[optind+2];

    anqa = anqfits_open(afn);
    if (!anqa) {
        fprintf(stderr, "Failed to open file \"%s\": %s.\n", afn, strerror(errno));
        exit(-1);
    }

    anqb = anqfits_open(bfn);
    if (!anqb) {
        fprintf(stderr, "Failed to open file \"%s\": %s.\n", bfn, strerror(errno));
        exit(-1);
    }

    atable = anqfits_get_table(anqa, 1);
    btable = anqfits_get_table(anqb, 1);

    if (!atable) {
        fprintf(stderr, "Failed to read a FITS table from ext 1 of file \"%s\".\n", afn);
        exit(-1);
    }
    if (!btable) {
        fprintf(stderr, "Failed to read a FITS table from ext 1 of file \"%s\".\n", bfn);
        exit(-1);
    }

    if (atable->tab_t != QFITS_BINTABLE) {
        fprintf(stderr, "Extension 1 of file \"%s\" doesn't contain a BINTABLE.\n", afn);
        exit(-1);
    }
    if (btable->tab_t != QFITS_BINTABLE) {
        fprintf(stderr, "Extension 1 of file \"%s\" doesn't contain a BINTABLE.\n", bfn);
        exit(-1);
    }

    if (atable->nr != btable->nr) {
        fprintf(stderr, "Input tables must have the same number of rows: %i vs %i.\n",
                atable->nr, btable->nr);
        exit(-1);
    }

    for (i=0; i<atable->nc; i++) {
        for (j=0; j<btable->nc; j++) {
            if (strcmp(atable->col[i].tlabel, btable->col[j].tlabel) == 0) {
                fprintf(stderr, "Input tables both have a column named \"%s\".\n", atable->col[i].tlabel);
                if (!suffix) {
                    exit(-1);
                }
                sprintf(btable->col[j].tlabel, "%s%s", atable->col[i].tlabel, suffix);
                printf("Changed name to \"%s\"\n", btable->col[j].tlabel);
            }
        }
    }

    afid = fopen(afn, "rb");
    if (!afid) {
        fprintf(stderr, "Failed to open file \"%s\": %s.\n", afn, strerror(errno));
        exit(-1);
    }
    bfid = fopen(bfn, "rb");
    if (!bfid) {
        fprintf(stderr, "Failed to open file \"%s\": %s.\n", bfn, strerror(errno));
        exit(-1);
    }

    aoff  = anqfits_data_start(anqa, 1);
    boff  = anqfits_data_start(anqb, 1);

    if (fseek(afid, aoff, SEEK_SET)) {
        fprintf(stderr, "Failed to seek to start of data in file \"%s\": %s\n", afn, strerror(errno));
        exit(-1);
    }
    if (fseek(bfid, boff, SEEK_SET)) {
        fprintf(stderr, "Failed to seek to start of data in file \"%s\": %s\n", bfn, strerror(errno));
        exit(-1);
    }

    outtable = qfits_table_new(outfn, QFITS_BINTABLE, atable->tab_w + btable->tab_w,
                               atable->nc + btable->nc, atable->nr);
    // copy column descriptions
    memcpy(outtable->col, atable->col, atable->nc * sizeof(qfits_col));
    memcpy(outtable->col + atable->nc, btable->col, btable->nc * sizeof(qfits_col));

    tablehdr = qfits_table_ext_header_default(outtable);
    if (!tablehdr) {
        fprintf(stderr, "Failed to create FITS table header.\n");
        exit(-1);
    }

    buffer = malloc(MAX(atable->tab_w, btable->tab_w));
    if (!buffer) {
        fprintf(stderr, "Failed to malloc buffer.\n");
        exit(-1);
    }

    outfid = fopen(outfn, "wb");
    if (!outfid) {
        fprintf(stderr, "Failed to open file \"%s\": %s.\n", outfn, strerror(errno));
        exit(-1);
    }

    hdr = anqfits_get_header2(afn, 0);
    if (!hdr) {
        fprintf(stderr, "Failed to read primary header from \"%s\".\n", afn);
        exit(-1);
    }
    BOILERPLATE_ADD_FITS_HEADERS(hdr);
    fits_add_long_history(hdr, "This file was created by the program \"%s\" by "
                          "merging columns from the input files \"%s\" and \"%s\"."
                          , args[0], afn, bfn);
    if (qfits_header_dump(hdr, outfid) ||
        qfits_header_dump(tablehdr, outfid)) {
        fprintf(stderr, "Failed to write headers.\n");
        exit(-1);
    }

    qfits_header_destroy(hdr);
    qfits_header_destroy(tablehdr);

    for (i=0; i<atable->nr; i++) {
        if (fread(buffer, 1, atable->tab_w, afid) != atable->tab_w) {
            fprintf(stderr, "Failed to read row %i from table \"%s\": %s\n", i, afn, strerror(errno));
            exit(-1);
        }
        if (fwrite(buffer, 1, atable->tab_w, outfid) != atable->tab_w) {
            fprintf(stderr, "Failed to write row %i: %s\n", i, strerror(errno));
            exit(-1);
        }
        if (fread(buffer, 1, btable->tab_w, bfid) != btable->tab_w) {
            fprintf(stderr, "Failed to read row %i from table \"%s\": %s\n", i, bfn, strerror(errno));
            exit(-1);
        }
        if (fwrite(buffer, 1, btable->tab_w, outfid) != btable->tab_w) {
            fprintf(stderr, "Failed to write row %i: %s\n", i, strerror(errno));
            exit(-1);
        }
    }
    free(buffer);

    if (fits_pad_file(outfid)) {
        fprintf(stderr, "Failed to zero-pad file.\n");
        exit(-1);
    }

    if (fclose(outfid)) {
        fprintf(stderr, "Failed to close output file: %s\n", strerror(errno));
        exit(-1);
    }

    fclose(afid);
    fclose(bfid);

    anqfits_close(anqa);
    anqfits_close(anqb);

    qfits_table_close(atable);
    qfits_table_close(btable);
    qfits_table_close(outtable);

    return 0;
}
