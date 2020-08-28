/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "startree.h"
#include "fitstable.h"
#include "boilerplate.h"
#include "errors.h"
#include "log.h"
#include "fitsioutils.h"

const char* OPTIONS = "hvL:d:t:bsSci:o:R:D:PTkn:";

void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "     -i <input-fits-catalog-name>\n"
           "     -o <output-star-kdtree-name>\n"
           "    [-R <ra-column-name>]: name of RA in FITS table (default RA)\n"
           "    [-D <dec-column-name>]: name of DEC in FITS table (default DEC)\n"
           "    [-b]: build bounding boxes (default: splitting planes)\n"
           "    [-L Nleaf]: number of points in a kdtree leaf node (default 25)\n"
           "    [-t  <tree type>]:  {double,float,u32,u16}, default u32.\n"
           "    [-d  <data type>]:  {double,float,u32,u16}, default u32.\n"
           "    [-S]: include separate splitdim array\n"
           "    [-c]: run kdtree_check on the resulting tree\n"
           "    [-P]: unpermute tree + tag-along data\n"
           "    [-T]: write tag-along table as first extension HDU\n"
           "    [-k]: keep RA,Dec columns in tag-along table\n"
           "    [-n <name>]: kd-tree name (default \"stars\")\n"
           "    [-v]: +verbose\n"
           "\n", progname);
}


int main(int argc, char *argv[]) {
    int argidx, argchar;
    startree_t* starkd;
    fitstable_t* cat;
    fitstable_t* tag;
    int Nleaf = 0;
    char* skdtfn = NULL;
    char* catfn = NULL;
    char* progname = argv[0];
    char* racol = NULL;
    char* deccol = NULL;
    int loglvl = LOG_MSG;
    char* treename = NULL;

    int datatype = 0;
    int treetype = 0;
    int buildopts = 0;
    anbool checktree = FALSE;
    anbool unpermute = FALSE;
    anbool remove_radec = TRUE;
    u32* perm = NULL;
    anbool tagalong_first = FALSE;
    
    if (argc <= 2) {
        printHelp(progname);
        return 0;
    }

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'T':
            tagalong_first = TRUE;
            break;
        case 'n':
            treename = optarg;
            break;
        case 'k':
            remove_radec = FALSE;
            break;
        case 'P':
            unpermute = TRUE;
            break;
        case 'R':
            racol = optarg;
            break;
        case 'D':
            deccol = optarg;
            break;
        case 'c':
            checktree = TRUE;
            break;
        case 'L':
            Nleaf = (int)strtoul(optarg, NULL, 0);
            break;
        case 'i':
            catfn = optarg;
            break;
        case 'o':
            skdtfn = optarg;
            break;
        case 't':
            treetype = kdtree_kdtype_parse_tree_string(optarg);
            break;
        case 'd':
            datatype = kdtree_kdtype_parse_data_string(optarg);
            break;
        case 'b':
            buildopts |= KD_BUILD_BBOX;
            break;
        case 's':
            buildopts |= KD_BUILD_SPLIT;
            break;
        case 'S':
            buildopts |= KD_BUILD_SPLITDIM;
            break;
        case 'v':
            loglvl++;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        case 'h':
            printHelp(progname);
            return 0;
        default:
            return -1;
        }

    if (optind < argc) {
        for (argidx = optind; argidx < argc; argidx++)
            fprintf (stderr, "Non-option argument %s\n", argv[argidx]);
        printHelp(progname);
        exit(-1);
    }

    if (!(catfn && skdtfn)) {
        printHelp(progname);
        exit(-1);
    }

    log_init(loglvl);
    fits_use_error_system();

    logmsg("Building star kdtree: reading %s, writing to %s\n", catfn, skdtfn);

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
    if (checktree) {
        logverb("Checking tree...\n");
        if (kdtree_check(starkd->tree)) {
            ERROR("kdtree_check failed!");
            exit(-1);
        }
    }

    if (treename) {
        free(starkd->tree->name);
        starkd->tree->name = strdup(treename);
    }

    if (unpermute) {
        perm = starkd->tree->perm;
        starkd->tree->perm = NULL;
    }

    if (tagalong_first) {
        logmsg("Writing tag-along data...\n");
        tag = fitstable_open_for_writing(skdtfn);
        if (fitstable_write_primary_header(tag)) {
            ERROR("Failed to write primary header");
            exit(-1);
        }
        if (startree_write_tagalong_table(cat, tag, racol, deccol,
                                          (int*)perm, remove_radec)) {
            ERROR("Failed to write tag-along table");
            exit(-1);
        }
        if (fitstable_close(tag)) {
            ERROR("Failed to close tag-along data");
            exit(-1);
        }
        // Append kd-tree
        logverb("Appending kd-tree structure...\n");
        FILE* fid = fopen(skdtfn, "r+b");
        if (!fid) {
            SYSERROR("Failed to open startree output file to append kd-tree: %s", skdtfn);
            exit(-1);
        }
        if (fseeko(fid, 0, SEEK_END)) {
            SYSERROR("Failed to seek to the end of the startree file to append kd-tree: %s", skdtfn);
            exit(-1);
        }
        off_t off = ftello(fid);
        printf("Offset to write starkd: %lu\n", (unsigned long)off);

        if (startree_append_to(starkd, fid)) {
            ERROR("Failed to append star kdtree");
            exit(-1);
        }
        startree_close(starkd);
        if (fclose(fid)) {
            SYSERROR("Failed to close star kdtree file after appending tree\n");
            exit(-1);
        }
    } else {
        if (startree_write_to_file(starkd, skdtfn)) {
            ERROR("Failed to write star kdtree");
            exit(-1);
        }
        startree_close(starkd);

        // Append tag-along table.
        logmsg("Writing tag-along data...\n");
        tag = fitstable_open_for_appending(skdtfn);

        if (startree_write_tagalong_table(cat, tag, racol, deccol,
                                          (int*)perm, remove_radec)) {
            ERROR("Failed to write tag-along table");
            exit(-1);
        }

        if (fitstable_close(tag)) {
            ERROR("Failed to close tag-along data");
            exit(-1);
        }
    }
    fitstable_close(cat);
    return 0;
}


