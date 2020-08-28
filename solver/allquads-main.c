/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>

#include "starutil.h"
#include "codefile.h"
#include "mathutil.h"
#include "quadfile.h"
#include "kdtree.h"
#include "fitsioutils.h"
#include "starkd.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "quad-utils.h"
#include "quad-builder.h"
#include "allquads.h"
#include "starkd.h"
#include "startree.h"
#include "kdtree.h"
#include "codekd.h"
#include "codetree.h"
#include "unpermute-quads.h"
#include "unpermute-stars.h"
#include "index.h"
#include "merge-index.h"

//#include "build-index.h"

const char* OPTIONS = "hi:o:u:l:d:I:v";

static void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "      -i <input-filename>    (FITS reference catalog)\n"
           "      -o <index-filename>    (Index file output filename)\n"
           "     [-u <scale>]    upper bound of quad scale (arcmin)\n"
           "     [-l <scale>]    lower bound of quad scale (arcmin)\n"
           "     [-d <dimquads>] number of stars in a \"quad\".\n"
           "     [-I <unique-id>] set the unique ID of this index\n\n"
           "\nReads skdt, writes {code, quad}.\n\n"
           , progname);
}


int main(int argc, char** argv) {
    int argchar;
    allquads_t* aq;
    int loglvl = LOG_MSG;
    int i, N;
    char* catfn = NULL;

    startree_t* starkd;
    fitstable_t* cat;

    char* racol = NULL;
    char* deccol = NULL;
    int datatype = KDT_DATA_DOUBLE;
    int treetype = KDT_TREE_DOUBLE;
    int buildopts = 0;
    int Nleaf = 0;

    char* indexfn = NULL;
    //index_params_t* p;

    aq = allquads_init();
    aq->skdtfn = "allquads.skdt";
    aq->codefn = "allquads.code";
    aq->quadfn = "allquads.quad";

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 'd':
            aq->dimquads = atoi(optarg);
            break;
        case 'I':
            aq->id = atoi(optarg);
            break;
        case 'h':
            print_help(argv[0]);
            exit(0);
        case 'i':
            catfn = optarg;
            break;
        case 'o':
            indexfn = optarg;
            break;
        case 'u':
            aq->quad_d2_upper = arcmin2distsq(atof(optarg));
            aq->use_d2_upper = TRUE;
            break;
        case 'l':
            aq->quad_d2_lower = arcmin2distsq(atof(optarg));
            aq->use_d2_lower = TRUE;
            break;
        default:
            return -1;
        }

    log_init(loglvl);

    if (!catfn || !indexfn) {
        printf("Specify in & out filenames, bonehead!\n");
        print_help(argv[0]);
        exit( -1);
    }

    if (optind != argc) {
        print_help(argv[0]);
        printf("\nExtra command-line args were given: ");
        for (i=optind; i<argc; i++) {
            printf("%s ", argv[i]);
        }
        printf("\n");
        exit(-1);
    }

    if (!aq->id)
        logmsg("Warning: you should set the unique-id for this index (with -I).\n");

    if (aq->dimquads > DQMAX) {
        ERROR("Quad dimension %i exceeds compiled-in max %i.\n", aq->dimquads, DQMAX);
        exit(-1);
    }
    aq->dimcodes = dimquad2dimcode(aq->dimquads);

    // Read reference catalog, write star kd-tree
    logmsg("Building star kdtree: reading %s, writing to %s\n", catfn, aq->skdtfn);
    
    logverb("Reading star catalogue...");
    cat = fitstable_open(catfn);
    if (!cat) {
        ERROR("Couldn't read catalog");
        exit(-1);
    }
    logmsg("Got %i stars\n", fitstable_nrows(cat));
    starkd = startree_build(cat, racol, deccol, datatype, treetype,
                            buildopts, Nleaf, argv, argc);
    if (!starkd) {
        ERROR("Failed to create star kdtree");
        exit(-1);
    }

    logmsg("Star kd-tree contains %i data points in dimension %i\n",
           startree_N(starkd), startree_D(starkd));
    N = startree_N(starkd);
    for (i=0; i<N; i++) {
        double ra,dec;
        int ok;
        ok = startree_get_radec(starkd, i, &ra, &dec);
        logmsg("  data %i: ok %i, RA,Dec %g, %g\n", i, ok, ra, dec);
    }

    if (startree_write_to_file(starkd, aq->skdtfn)) {
        ERROR("Failed to write star kdtree");
        exit(-1);
    }
    startree_close(starkd);
    fitstable_close(cat);
    logmsg("Wrote star kdtree to %s\n", aq->skdtfn);

    logmsg("Running allquads...\n");
    if (allquads_open_outputs(aq)) {
        exit(-1);
    }

    if (allquads_create_quads(aq)) {
        exit(-1);
    }

    if (allquads_close(aq)) {
        exit(-1);
    }

    logmsg("allquads: wrote %s, %s\n", aq->quadfn, aq->codefn);

    // build-index:
    //build_index_defaults(&p);

    // codetree
    /*
     if (step_codetree(p, codes, &codekd,
     codefn, &ckdtfn, tempfiles))
     return -1;
     */
    char* ckdtfn=NULL;
    char* tempdir = NULL;

    ckdtfn = create_temp_file("ckdt", tempdir);
    logmsg("Creating code kdtree, reading %s, writing to %s\n", aq->codefn, ckdtfn);
    if (codetree_files(aq->codefn, ckdtfn, 0, 0, 0, 0, argv, argc)) {
        ERROR("codetree failed");
        return -1;
    }

    char* skdt2fn=NULL;
    char* quad2fn=NULL;

    // unpermute-stars
    logmsg("Unpermute-stars...\n");
    skdt2fn = create_temp_file("skdt2", tempdir);
    quad2fn = create_temp_file("quad2", tempdir);

    logmsg("Unpermuting stars from %s and %s to %s and %s\n",
           aq->skdtfn, aq->quadfn, skdt2fn, quad2fn);
    if (unpermute_stars_files(aq->skdtfn, aq->quadfn, skdt2fn, quad2fn,
                              TRUE, FALSE, argv, argc)) {
        ERROR("Failed to unpermute-stars");
        return -1;
    }

    allquads_free(aq);

    // unpermute-quads
    /*
     if (step_unpermute_quads(p, quads2, codekd, &quads3, &codekd2,
     quad2fn, ckdtfn, &quad3fn, &ckdt2fn, tempfiles))
     return -1;
     */
    char* quad3fn=NULL;
    char* ckdt2fn=NULL;

    ckdt2fn = create_temp_file("ckdt2", tempdir);
    quad3fn = create_temp_file("quad3", tempdir);
    logmsg("Unpermuting quads from %s and %s to %s and %s\n",
           quad2fn, ckdtfn, quad3fn, ckdt2fn);
    if (unpermute_quads_files(quad2fn, ckdtfn,
                              quad3fn, ckdt2fn, argv, argc)) {
        ERROR("Failed to unpermute-quads");
        return -1;
    }

    // index
    /*
     if (step_merge_index(p, codekd2, quads3, starkd2, p_index,
     ckdt2fn, quad3fn, skdt2fn, indexfn))
     return -1;
     */
    quadfile_t* quad;
    codetree_t* code;
    startree_t* star;

    logmsg("Merging %s and %s and %s to %s\n",
           quad3fn, ckdt2fn, skdt2fn, indexfn);
    if (merge_index_open_files(quad3fn, ckdt2fn, skdt2fn,
                               &quad, &code, &star)) {
        ERROR("Failed to open index files for merging");
        return -1;
    }
    if (merge_index(quad, code, star, indexfn)) {
        ERROR("Failed to write merged index");
        return -1;
    }
    codetree_close(code);
    startree_close(star);
    quadfile_close(quad);

    printf("Done.\n");

    free(ckdtfn);
    free(skdt2fn);
    free(quad2fn);
    free(ckdt2fn);
    free(quad3fn);

    return 0;
}

