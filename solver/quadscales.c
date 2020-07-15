/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "quadfile.h"
#include "kdtree.h"
#include "starutil.h"
#include "mathutil.h"
#include "bl.h"
#include "histogram.h"
#include "starkd.h"
#include "boilerplate.h"

#define OPTIONS "h"


void print_help(char* progname)
{
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "Usage: %s\n"
            "   [-h]: help\n"
            "   [-n <number of histogram bins>]  (default 100)\n"
            "   <base-name> [<base-name> ...]\n\n"
            "Requires both skdt and quad files.  Outputs Matlab literals.\n"
            "\n", progname);
}

int main(int argc, char** args) {
    int argchar;
    char* basename;
    char* fn;
    quadfile* qf;
    startree_t* skdt = NULL;
    int i;
    int Nbins = 100;
    histogram* sumhist = NULL;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'n':
            Nbins = atoi(optarg);
            break;
        case 'h':
            print_help(args[0]);
            exit(0);
        }

    if (optind == argc) {
        print_help(args[0]);
        exit(-1);
    }

    for (; optind<argc; optind++) {
        int lastgrass = 0;
        histogram* hist;
        char fixed_basename[256];
        int dimquads;

        basename = args[optind];
        fprintf(stderr, "Reading files with basename %s\n", basename);

        snprintf(fixed_basename, 256, "%s", basename);
        for (i=0; i<strlen(fixed_basename); i++) {
            if (fixed_basename[i] == '-')
                fixed_basename[i] = '_';
        }

        asprintf_safe(&fn, "%s.quad.fits", basename);
        fprintf(stderr, "Opening quad file %s...\n", fn);
        qf = quadfile_open(fn);
        if (!qf) {
            fprintf(stderr, "Failed to open quad file %s.\n", fn);
            continue;
        }
        free(fn);

        asprintf_safe(&fn, "%s.skdt.fits", basename);
        fprintf(stderr, "Opening skdt file %s...\n", fn);
        skdt = startree_open(fn);
        if (!skdt) {
            fprintf(stderr, "Failed to read star kdtree %s.\n", fn);
            continue;
        }
        free(fn);

        hist = histogram_new_nbins(qf->index_scale_lower,
                                   qf->index_scale_upper, Nbins);

        if (!sumhist)
            sumhist = histogram_new_nbins(qf->index_scale_lower,
                                          qf->index_scale_upper, Nbins);
        else {
            if ((sumhist->min != hist->min) ||
                (sumhist->binsize != hist->binsize))
                fprintf(stderr, "Warning, the sum-total histogram's range doesn't match this index's range.\n");
        }

        fprintf(stderr, "Reading %i quads...\n", qf->numquads);

        dimquads = quadfile_dimquads(qf);
        assert(dimquads >= 2);

        for (i=0; i<qf->numquads; i++) {
            unsigned int stars[dimquads];
            int grass;
            double xyzA[3];
            double xyzB[3];
            int A, B;
            double rad;

            grass = (i * 80 / qf->numquads);
            if (grass != lastgrass) {
                fprintf(stderr, ".");
                fflush(stderr);
                lastgrass = grass;
            }

            quadfile_get_stars(qf, i, stars);

            A = stars[0];
            B = stars[1];
            startree_get(skdt, A, xyzA);
            startree_get(skdt, B, xyzB);
            rad = distsq2arc(distsq(xyzA, xyzB, 3));

            histogram_add(hist, rad);
            histogram_add(sumhist, rad);
        }
        fprintf(stderr, "\n");

        printf("%s = ", fixed_basename);
        histogram_print_matlab(hist, stdout);
        printf(";\n");
        printf("%s_bins = ", fixed_basename);
        histogram_print_matlab_bin_centers(hist, stdout);
        printf(";\n");

        histogram_free(hist);

        startree_close(skdt);
        quadfile_close(qf);
    }

    printf("qs_sum = ");
    histogram_print_matlab(sumhist, stdout);
    printf(";\n");
    printf("qs_sum_bins = ");
    histogram_print_matlab_bin_centers(sumhist, stdout);
    printf(";\n");

    histogram_free(sumhist);

    return 0;
}
