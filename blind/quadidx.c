/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 Quadidx: create .qidx files from .quad files.

 A .quad file lists, for each quad, the stars comprising the quad.
 A .qidx file lists, for each star, the quads that star is a member of.

 Input: .quad
 Output: .qidx
 */

#include <string.h>

#include "starutil.h"
#include "quadfile.h"
#include "qidxfile.h"
#include "bl.h"
#include "fitsioutils.h"
#include "boilerplate.h"
#include "log.h"
#include "errors.h"

static const char* OPTIONS = "hFi:o:cv";

static void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s -i <quad input> -o <qidx output>\n"
           "\n"
           "    [-c]: run quadfile_check()\n"
           "    [-v]: add to verboseness\n"
           "\n", progname);
}


int main(int argc, char *argv[]) {
    char* progname = argv[0];
    int argidx, argchar;
    char *idxfname = NULL;
    char *quadfname = NULL;
    il** quadlist;
    quadfile_t* quads;
    qidxfile* qidx;
    int q;
    int i;
    int numused;
    qfits_header* quadhdr;
    qfits_header* qidxhdr;
    int dimquads;
    anbool check = FALSE;
    int loglvl = LOG_MSG;
	
    if (argc <= 2) {
        printHelp(progname);
        exit(-1);
    }

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'c':
            check = TRUE;
            break;
        case 'v':
            loglvl++;
            break;
        case 'i':
            quadfname = optarg;
            break;
        case 'o':
            idxfname = optarg;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            exit(-1);
        default:
            return (OPT_ERR);
        }

    log_init(loglvl);

    if (optind < argc) {
        for (argidx = optind; argidx < argc; argidx++)
            fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
        printHelp(progname);
        exit(-1);
    }

    logmsg("quadidx: indexing quads in \"%s\"...\n", quadfname);
    logmsg("will write to file \"%s\".\n", idxfname);

    quads = quadfile_open(quadfname);
    if (!quads) {
        ERROR("Couldn't open quads file \"%s\"", quadfname);
        exit(-1);
    }
    logmsg("%u quads, %u stars.\n", quads->numquads, quads->numstars);

    if (check) {
        logmsg("Running quadfile_check()...\n");
        if (quadfile_check(quads)) {
            ERROR("quadfile_check() failed");
            exit(-1);
        }
        logmsg("Check passed.\n");
    }

    quadlist = calloc(quads->numstars, sizeof(il*));
    if (!quadlist) {
        SYSERROR("Failed to allocate list of quad contents");
        exit(-1);
    }

    dimquads = quadfile_dimquads(quads);
    for (q=0; q<quads->numquads; q++) {
        unsigned int inds[dimquads];
        quadfile_get_stars(quads, q, inds);

        // append this quad index to the lists of each of its stars.
        for (i=0; i<dimquads; i++) {
            il* list;
            int starind = inds[i];
            list = quadlist[starind];
            // create the list if necessary
            if (!list) {
                list = il_new(10);
                quadlist[starind] = list;
            }
            il_append(list, q);
        }
    }
	
    // first count numused:
    // how many stars are members of quads.
    numused = 0;
    for (i=0; i<quads->numstars; i++) {
        il* list = quadlist[i];
        if (!list) continue;
        numused++;
    }
    logmsg("%u stars used\n", numused);

    qidx = qidxfile_open_for_writing(idxfname, quads->numstars, quads->numquads);
    if (!qidx) {
        logmsg("Couldn't open outfile qidx file %s.\n", idxfname);
        exit(-1);
    }

    quadhdr = quadfile_get_header(quads);
    qidxhdr = qidxfile_get_header(qidx);

    an_fits_copy_header(quadhdr, qidxhdr, "INDEXID");
    an_fits_copy_header(quadhdr, qidxhdr, "HEALPIX");

    BOILERPLATE_ADD_FITS_HEADERS(qidxhdr);
    qfits_header_add(qidxhdr, "HISTORY", "This file was created by the program \"quadidx\".", NULL, NULL);
    qfits_header_add(qidxhdr, "HISTORY", "quadidx command line:", NULL, NULL);
    fits_add_args(qidxhdr, argv, argc);
    qfits_header_add(qidxhdr, "HISTORY", "(end of quadidx command line)", NULL, NULL);

    qfits_header_add(qidxhdr, "HISTORY", "** History entries copied from the input file:", NULL, NULL);
    fits_copy_all_headers(quadhdr, qidxhdr, "HISTORY");
    qfits_header_add(qidxhdr, "HISTORY", "** End of history entries.", NULL, NULL);

    if (qidxfile_write_header(qidx)) {
        logmsg("Couldn't write qidx header (%s).\n", idxfname);
        exit(-1);
    }

    for (i=0; i<quads->numstars; i++) {
        int thisnumq;
        //int thisstar;
        int* stars; // bad variable name - list of quads this star is in.
        il* list = quadlist[i];
        if (list) {
            thisnumq = (uint)il_size(list);
            stars = malloc(thisnumq * sizeof(uint));
            il_copy(list, 0, thisnumq, (int*)stars);
        } else {
            thisnumq = 0;
            stars = NULL;
        }
        //thisstar = i;

        if (qidxfile_write_star(qidx,  stars, thisnumq)) {
            logmsg("Couldn't write star to qidx file (%s).\n", idxfname);
            exit(-1);
        }

        if (list) {
            free(stars);
            il_free(list);
            quadlist[i] = NULL;
        }
    }
    free(quadlist);
    quadfile_close(quads);

    if (qidxfile_close(qidx)) {
        logmsg("Failed to close qidx file.\n");
        exit(-1);
    }

    logmsg("  done.\n");
    return 0;
}

