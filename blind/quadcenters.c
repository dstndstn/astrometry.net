/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "index.h"
#include "quadfile.h"
#include "kdtree.h"
#include "starutil.h"
#include "mathutil.h"
#include "fitsioutils.h"
#include "bl.h"
#include "starkd.h"
#include "boilerplate.h"
#include "rdlist.h"

#define OPTIONS "hr:R"


void print_help(char* progname)
{
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "Usage: %s\n"
            "   -r <rdls-output-file>\n"
            "   [-R]: add quad radius columns to the RDLS file.\n"
            "   [-h]: help\n"
            "   <base-name> [<base-name> ...]\n\n"
            "Reads index (.quad and .skdt or merged index) files.\n"
            "Writes an RDLS containing the quad centers (midpoints of AB), one field per input file.\n\n",
            progname);
}

int main(int argc, char** args) {
    int argchar;
    char* basename;
    char* outfn = NULL;
    index_t* index;
    quadfile* qf;
    rdlist_t* rdls;
    startree_t* skdt = NULL;
    anbool addradius = FALSE;
    int i;
    int radcolumn = -1;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'R':
            addradius = TRUE;
            break;
        case 'r':
            outfn = optarg;
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        }

    if (!outfn || (optind == argc)) {
        print_help(args[0]);
        exit(-1);
    }

    rdls = rdlist_open_for_writing(outfn);
    if (!rdls) {
        fprintf(stderr, "Failed to open RDLS file %s for output.\n", outfn);
        exit(-1);
    }
    if (rdlist_write_primary_header(rdls)) {
        fprintf(stderr, "Failed to write RDLS header.\n");
        exit(-1);
    }

    if (addradius) {
        radcolumn = rdlist_add_tagalong_column(rdls, fitscolumn_double_type(),
                                               1, fitscolumn_double_type(),
                                               "QUADRADIUS", "deg");
    }

    for (; optind<argc; optind++) {
        //int Nstars;
        int dimquads;

        basename = args[optind];
        printf("Reading files with basename %s\n", basename);

        index = index_load(basename, 0, NULL);
        if (!index) {
            fprintf(stderr, "Failed to read index with base name \"%s\"\n", basename);
            exit(-1);
        }

        qf = index->quads;
        skdt = index->starkd;
        //Nstars = startree_N(skdt);
        
        if (rdlist_write_header(rdls)) {
            fprintf(stderr, "Failed to write new RDLS field header.\n");
            exit(-1);
        }

        dimquads = quadfile_dimquads(qf);

        printf("Reading quads...\n");
        for (i=0; i<qf->numquads; i++) {
            unsigned int stars[dimquads];
            double axyz[3], bxyz[3];
            double midab[3];
            double radec[2];
            if (!(i % 200000)) {
                printf(".");
                fflush(stdout);
            }
            quadfile_get_stars(qf, i, stars);
            startree_get(skdt, stars[0], axyz);
            startree_get(skdt, stars[1], bxyz);
            star_midpoint(midab, axyz, bxyz);
            xyzarr2radecdegarr(midab, radec);

            if (rdlist_write_one_radec(rdls, radec[0], radec[1])) {
                fprintf(stderr, "Failed to write a RA,Dec entry.\n");
                exit(-1);
            }

            if (addradius) {
                double rad = arcsec2deg(distsq2arcsec(distsq(midab, axyz, 3)));
                if (rdlist_write_tagalong_column(rdls, radcolumn, i, 1, &rad, 0)) {
                    fprintf(stderr, "Failed to write quad radius.\n");
                    exit(-1);
                }
            }
        }
        printf("\n");

        index_close(index);

        if (rdlist_fix_header(rdls)) {
            fprintf(stderr, "Failed to fix RDLS field header.\n");
            exit(-1);
        }

        rdlist_next_field(rdls);
    }

    if (rdlist_fix_primary_header(rdls) ||
        rdlist_close(rdls)) {
        fprintf(stderr, "Failed to close RDLS file.\n");
        exit(-1);
    }

    return 0;
}
