/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "healpix.h"
#include "starutil.h"
#include "rdlist.h"
#include "bl.h"

char* OPTIONS = "hqf:N:";

void printHelp(char* progname) {
    fprintf(stderr, "Usage: %s [options]\n"
            "   -f <rdls-file>\n"
            "   [-N nside]   (default 1)\n"
            "   [-h] print help msg\n"
            "   [-q] quiet mode\n",
            progname);
}


int main(int argc, char** args) {
    char* filename = NULL;
    int npoints;
    int i, j;
    int* healpixes;
    int argchar;
    char* progname = args[0];
    il** lists;
    anbool quiet = FALSE;
    rdlist* rdls;
    int Nside = 1;
    int N;

    while ((argchar = getopt (argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'N':
            Nside = atoi(optarg);
            break;
        case 'f':
            filename = optarg;
            break;
        case 'h':
            printHelp(progname);
            exit(0);
        case 'q':
            quiet = TRUE;
            break;
        case '?':
            fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        default:
            exit(-1);
        }

    if (!filename) {
        printHelp(progname);
        exit(-1);
    }

    fprintf(stderr, "Opening RDLS file %s...\n", filename);
    rdls = rdlist_open(filename);
    if (!rdls) {
        fprintf(stderr, "Failed to open RDLS file.\n");
        exit(-1);
    }

    N = 12 * Nside * Nside;

    healpixes = malloc(N * sizeof(int));
    lists     = calloc(N,  sizeof(il*));

    /*
     for (i=0; i<N; i++) {
     lists[i] = il_new(256);
     }
     */

    for (j=1; j<=rdls_n_fields(rdls); j++) {
        rd* points;

        points = rdlist_get_field(rdls, j);
        if (!points) {
            fprintf(stderr, "error reading field %i\n", j);
            break;
        }

        memset(healpixes, 0, N * sizeof(int));

        npoints = rd_size(points);

        for (i=0; i<npoints; i++) {
            double ra, dec;
            int hp;

            ra  = deg2rad(rd_refra (points, i));
            dec = deg2rad(rd_refdec(points, i));

            if (Nside > 1)
                hp = radectohealpix_nside(ra, dec, Nside);
            else
                hp = radectohealpix(ra, dec);
            if ((hp < 0) || (hp >= N)) {
                printf("hp=%i\n", hp);
                continue;
            }
            healpixes[hp] = 1;
        }
        if (!quiet) {
            printf("Field %i: healpixes  ", j);
            for (i=0; i<N; i++) {
                if (healpixes[i])
                    printf("%i  ", i);
            }
            printf("\n");
            fflush(stdout);
        }

        for (i=0; i<N; i++)
            if (healpixes[i]) {
                if (!lists[i])
                    lists[i] = il_new(256);
                il_append(lists[i], j);
            }

        free_rd(points);
    }

    for (i=0; i<N; i++) {
        int N;
        if (!lists[i]) 
            continue;
        printf("HP %i: ", i);
        N = il_size(lists[i]);
        for (j=0; j<N; j++)
            printf("%i ", il_get(lists[i], j));
        il_free(lists[i]);
        printf("\n");
    }

    free(lists);
    free(healpixes);

    rdlist_close(rdls);
    return 0;
}
