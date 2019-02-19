/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "healpix.h"
#include "starutil.h"
#include "mathutil.h"
#include "boilerplate.h"

#define OPTIONS "hN:mf:"

void print_help(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s\n"
           "     [-N <nside>]    (default 1)\n"
           "     [-m (to include a margin of one small healpixel)]\n"
           "     [-f <format>]: printf format for the output (default %%03i)\n"
           "     <hp> [<hp> ...]\n",
           progname);
}


int main(int argc, char** args) {
    int c;
    int Nside = 1;
    int HP;
    int optstart;
    anbool margin = FALSE;
    int* owned;
    int i;
    double hparea;
    char* format = "%03i";

    if (argc == 1) {
        print_help(args[0]);
        exit(0);
    }

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case '?':
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'f':
            format = optarg;
            break;
        case 'N':
            Nside = atoi(optarg);
            break;
        case 'm':
            margin = TRUE;
            break;
        }
    }
    optstart = optind;

    HP = 12 * Nside * Nside;
    fprintf(stderr, "Nside=%i, number of small healpixes=%i, margin=%c\n", Nside, HP, (margin?'Y':'N'));
    hparea = 4.0 * M_PI * square(rad2arcmin(1.0)) / (double)HP;
    fprintf(stderr, "Small healpix area = %g arcmin^2, length ~ %g arcmin.\n", hparea, sqrt(hparea));

    owned = malloc(HP * sizeof(int));
    for (optind=optstart; optind<argc; optind++) {
        int bighp = atoi(args[optind]);
        memset(owned, 0, HP * sizeof(int));
        // for each big healpix, find the set of small healpixes it owns
        // (including a bit of overlap)
        for (i=0; i<HP; i++) {
            int big, x, y;
            int nn, neigh[8], k;
            healpix_decompose_xy(i, &big, &x, &y, Nside);
            if (big != bighp)
                continue;
            owned[i] = 1;
            if (margin) {
                if (x == 0 || y == 0 || (x == Nside-1) || (y == Nside-1)) {
                    // add its neighbours.
                    nn = healpix_get_neighbours(i, neigh, Nside);
                    for (k=0; k<nn; k++)
                        owned[neigh[k]] = 1;
                }
            }
        }
        //printf("HP %i owns:\n", bighp);
        for (i=0; i<HP; i++) {
            if (owned[i]) {
                printf(format, i);
                printf(" ");
            }
        }
        printf("\n");
    }
    free(owned);

    return 0;
}
