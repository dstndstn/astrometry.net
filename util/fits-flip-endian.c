/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "an-bool.h"
#include "anqfits.h"
#include "bl.h"
#include "ioutils.h"
#include "errors.h"
#include "log.h"
#include "an-endian.h"

char* OPTIONS = "hi:o:e:s:";

void printHelp(char* progname) {
    fprintf(stderr, "%s -i <input-file>\n"
            "      -o <output-file>\n"
            "    [ -e <extension-number> -s <data-size-in-bytes> ] ...\n\n",
            progname);
}


int main(int argc, char *argv[]) {
    int argchar;
    char* infn = NULL;
    char* outfn = NULL;
    anbool tostdout = FALSE;
    FILE* fin = NULL;
    FILE* fout = NULL;
    il* exts;
    il* sizes;
    int i;
    char* progname = argv[0];
    int Next;
    anqfits_t* anq;

    exts = il_new(16);
    sizes = il_new(16);

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'e':
            il_append(exts, atoi(optarg));
            break;
        case 's':
            il_append(sizes, atoi(optarg));
            break;
        case 'i':
            infn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case '?':
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    log_init(LOG_MSG);

    if (!infn || !outfn || !il_size(exts) || (il_size(exts) != il_size(sizes))) {
        printHelp(progname);
        exit(-1);
    }

    if (infn) {
        fin = fopen(infn, "rb");
        if (!fin) {
            SYSERROR("Failed to open input file %s", infn);
            exit(-1);
        }
    }
    
    anq = anqfits_open(infn);
    if (!anq) {
        ERROR("Failed to open input file %s", infn);
        exit(-1);
    }
    Next = anqfits_n_ext(anq);
    if (Next == -1) {
        ERROR("Couldn't determine how many extensions are in file %s", infn);
        exit(-1);
    } else {
        logverb("File %s contains %i FITS extensions.\n", infn, Next);
    }

    for (i=0; i<il_size(exts); i++) {
        int e = il_get(exts, i);
        int s = il_get(sizes, i);
        if (e < 0 || e >= Next) {
            logerr("Extension %i is not valid: must be in [%i, %i]\n", e, 0, Next);
            exit(-1);
        }
        if (s != 2 && s != 4 && s != 8) {
            logerr("Invalid byte size %i: must be 2, 4, or 8.\n", s);
            exit(-1);
        }
    }

    if (!strcmp(outfn, "-"))
        tostdout = TRUE;

    if (tostdout)
        fout = stdout;
    else {
        fout = fopen(outfn, "wb");
        if (!fout) {
            SYSERROR("Failed to open output file %s", outfn);
            exit(-1);
        }
    }

    for (i=0; i<Next; i++) {
        int hdrstart, hdrlen, datastart, datalen;
        int ind;
        int size;
        ind = il_index_of(exts, i);
        if (ind == -1) {
            size = 0;
        } else {
            size = il_get(sizes, ind);
        }

        hdrstart = anqfits_header_start(anq, i);
        hdrlen   = anqfits_header_size (anq, i);
        datastart = anqfits_data_start(anq, i);
        datalen   = anqfits_data_size (anq, i);

        if (hdrlen) {
            if (pipe_file_offset(fin, hdrstart, hdrlen, fout)) {
                ERROR("Failed to write header for extension %i", i);
                exit(-1);
            }
        }
        if (!datalen)
            continue;

        if (size) {
            int Nitems = datalen / size;
            int j;
            char buf[size];
            logmsg("Extension %i: flipping words of length %i bytes.\n", i, size);
            for (j=0; j<Nitems; j++) {
                if (fread(buf, size, 1, fin) != 1) {
                    SYSERROR("Failed to read data element %i from extension %i", j, i);
                    exit(-1);
                }
                endian_swap(buf, size);
                if (fwrite(buf, size, 1, fout) != 1) {
                    SYSERROR("Failed to write data element %i to extension %i", j, i);
                    exit(-1);
                }
            }
        } else {
            logmsg("Extension %i: copying verbatim.\n", i);
            // passthrough
            if (pipe_file_offset(fin, datastart, datalen, fout)) {
                ERROR("Failed to write data for extension %i", i);
                exit(-1);
            }
        }
    }
    fclose(fin);
    anqfits_close(anq);
    if (!tostdout)
        fclose(fout);
    il_free(exts);
    il_free(sizes);
    return 0;
}
