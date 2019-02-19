/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include "an-bool.h"
#include "anqfits.h"
#include "bl.h"
#include "ioutils.h"
#include "log.h"
#include "fitsioutils.h"
#include "errors.h"

char* OPTIONS = "he:i:o:baDHMv";

void printHelp(char* progname) {
    fprintf(stderr, "%s    -i <input-file>\n"
            "      -o <output-file>\n"
            "      [-a]: write out ALL extensions; the output filename should be\n"
            "            a \"sprintf\" pattern such as  \"extension-%%04i\".\n"
            "      [-b]: print sizes and offsets in FITS blocks (of 2880 bytes)\n"
            "      [-M]: print sizes in megabytes (using floor(), not round()!)\n"
            "      [-D]: data blocks only\n"
            "      [-H]: header blocks only\n"
            "      -e <extension-number> ...\n"
            "      -v: +verbose\n\n",
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
    int i;
    char* progname = argv[0];
    anbool inblocks = FALSE;
    anbool inmegs = FALSE;
    int allexts = 0;
    int Next = -1;
    anbool dataonly = FALSE;
    anbool headeronly = FALSE;
    anqfits_t* anq = NULL;
    int loglvl = LOG_MSG;

    exts = il_new(16);

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 'D':
            dataonly = TRUE;
            break;
        case 'H':
            headeronly = TRUE;
            break;
        case 'a':
            allexts = 1;
            break;
        case 'b':
            inblocks = TRUE;
            break;
        case 'M':
            inmegs = TRUE;
            break;
        case 'e':
            il_append(exts, atoi(optarg));
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

    if (headeronly && dataonly) {
        fprintf(stderr, "Can't write data blocks only AND header blocks only!\n");
        exit(-1);
    }

    if (inblocks && inmegs) {
        fprintf(stderr, "Can't write sizes in FITS blocks and megabytes.\n");
        exit(-1);
    }

    fits_use_error_system();
    log_init(loglvl);
    log_to(stderr);
    errors_log_to(stderr);

    if (infn) {
        anq = anqfits_open(infn);
        if (!anq) {
            ERROR("Failed to open input file \"%s\"", infn);
            exit(-1);
        }
        Next = anqfits_n_ext(anq);
        fprintf(stderr, "File %s contains %i FITS extensions.\n", infn, Next);
    }

    if (infn && !outfn) {
        for (i=0; i<Next; i++) {
            off_t hdrstart, hdrlen, datastart, datalen;

            hdrstart  = anqfits_header_start(anq, i);
            hdrlen    = anqfits_header_size(anq, i);
            datastart = anqfits_data_start(anq, i);
            datalen   = anqfits_data_size(anq, i);

            if (inblocks) {
                off_t block = (off_t)FITS_BLOCK_SIZE;
                fprintf(stderr, "Extension %i : header start %zu , length %zu ; data start %zu , length %zu blocks.\n",
			i, (size_t)(hdrstart / block), (size_t)(hdrlen / block), (size_t)(datastart / block), (size_t)(datalen / block));
            } else if (inmegs) {
                off_t meg = 1024*1024;
                fprintf(stderr, "Extension %i : header start %zu , length %zu ; data start %zu , length %zu megabytes.\n",
			i, (size_t)(hdrstart/meg), (size_t)(hdrlen/meg), (size_t)(datastart/meg), (size_t)(datalen/meg));
            } else {
                fprintf(stderr, "Extension %i : header start %zu , length %zu ; data start %zu , length %zu .\n",
			i, (size_t)hdrstart, (size_t)hdrlen, (size_t)datastart, (size_t)datalen);
            }
        }
        anqfits_close(anq);
        exit(0);
    }

    if (!infn || !outfn || !(il_size(exts) || allexts)) {
        printHelp(progname);
        exit(-1);
    }

    if (!strcmp(outfn, "-")) {
        tostdout = TRUE;
        if (allexts) {
            fprintf(stderr, "Specify all extensions (-a) and outputting to stdout (-o -) doesn't make much sense...\n");
            exit(-1);
        }
    }

    if (infn) {
        fin = fopen(infn, "rb");
        if (!fin) {
            fprintf(stderr, "Failed to open input file %s: %s\n", infn, strerror(errno));
            exit(-1);
        }
    }

    if (tostdout)
        fout = stdout;
    else {
        if (allexts)
            for (i=0; i<Next; i++)
                il_append(exts, i);
        else {
            // open the (single) output file.
            fout = fopen(outfn, "wb");
            if (!fout) {
                fprintf(stderr, "Failed to open output file %s: %s\n", outfn, strerror(errno));
                exit(-1);
            }
        }
    }

    for (i=0; i<il_size(exts); i++) {
        off_t hdrstart, hdrlen, datastart, datalen;
        int ext = il_get(exts, i);

        if (allexts) {
            char fn[256];
            snprintf(fn, sizeof(fn), outfn, ext);
            fout = fopen(fn, "wb");
            if (!fout) {
                fprintf(stderr, "Failed to open output file %s: %s\n", fn, strerror(errno));
                exit(-1);
            }
        }

        hdrstart  = anqfits_header_start(anq, ext);
        hdrlen    = anqfits_header_size(anq, ext);
        datastart = anqfits_data_start(anq, ext);
        datalen   = anqfits_data_size(anq, ext);

        if (inblocks) {
            off_t block = (off_t)FITS_BLOCK_SIZE;
            fprintf(stderr, "Writing extension %i : header start %zu , length %zu ; data start %zu , length %zu blocks.\n",
                    ext, (size_t)(hdrstart / block), (size_t)(hdrlen / block), (size_t)(datastart / block), (size_t)(datalen / block));
        } else if (inmegs) {
            off_t meg = 1024*1024;
            fprintf(stderr, "Writing extension %i : header start %zu , length %zu ; data start %zu , length %zu megabytes.\n",
                    ext, (size_t)(hdrstart/meg), (size_t)(hdrlen/meg), (size_t)(datastart/meg), (size_t)(datalen/meg));
        } else {
            fprintf(stderr, "Writing extension %i : header start %zu , length %zu ; data start %zu , length %zu .\n",
                    ext, (size_t)hdrstart, (size_t)hdrlen, (size_t)datastart, (size_t)datalen);
        }

        if (hdrlen && !dataonly) {
            if (pipe_file_offset(fin, hdrstart, hdrlen, fout)) {
                fprintf(stderr, "Failed to write header for extension %i: %s\n", ext, strerror(errno));
                exit(-1);
            }
        }
        if (datalen && !headeronly) {
            if (pipe_file_offset(fin, datastart, datalen, fout)) {
                fprintf(stderr, "Failed to write data for extension %i: %s\n", ext, strerror(errno));
                exit(-1);
            }
        }

        if (allexts)
            if (fclose(fout)) {
                fprintf(stderr, "Failed to close output file: %s\n", strerror(errno));
                exit(-1);
            }
    }

    fclose(fin);
    if (!allexts && !tostdout)
        fclose(fout);
    il_free(exts);
    anqfits_close(anq);
    return 0;
}
