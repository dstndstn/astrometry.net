/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/*
 Write Matlab figures to demonstrate the correctness of the Healpix code.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "healpix.h"
#include "starutil.h"
#include "mathutil.h"

#define OPTIONS "hN:nM:F:rRe"

void print_help(char* progname) {
    printf("usage:\n\n"
           "%s\n"
           "  [-N <nside>]  (default 1)\n"
           "  [-n]: draw lines to show neighbours\n"
           "  [-M <marker-size>]: change the size of the circles on the endpoints of the neighbour lines.\n"
           "  [-F <font-size>]\n"
           "  [-r]: show RING index\n"
           "  [-R]: show decomposed RING index (ring number + longitude index)\n"
           "  [-e]: show NESTED index\n"
           "\n", progname);
}

enum modes {
    MODE_XY,
    MODE_XY_D,
    MODE_RING,
    MODE_RING_D,
    MODE_NESTED
};

int main(int argc, char** args) {
    int c;
    int Nside = 1;
    int HP, hp;
    int i;
    double* radecs;
    double markersize = 20.0;
    double fontsize = 10.0;
    anbool do_neighbours = FALSE;

    int mode = MODE_XY;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case '?':
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'N':
            Nside = atoi(optarg);
            break;
        case 'n':
            do_neighbours = TRUE;
            break;
        case 'M':
            markersize = atof(optarg);
            break;
        case 'F':
            fontsize = atof(optarg);
            break;
        case 'r':
            mode = MODE_RING;
            break;
        case 'R':
            mode = MODE_RING_D;
            break;
        case 'e':
            mode = MODE_NESTED;
            break;
        }
    }

    HP = 12 * Nside * Nside;

    printf("Nside=%i;\n", Nside);

    radecs = malloc(HP * 2 * sizeof(double));
	
    for (hp=0; hp<HP; hp++) {
        double xyz[3];
        double ra, dec;
        healpix_to_xyzarr(hp, Nside, 0.5, 0.5, xyz);
        xyzarr2radec(xyz, &ra, &dec);
        radecs[2*hp] = ra;
        radecs[2*hp+1] = dec;
    }

    printf("figure(1);\n");
    printf("clf;\n");
    printf("xmargin=0.5; ymargin=0.1;\n");
    printf("axis([0-xmargin, 2*pi+xmargin, -pi/2-ymargin, pi/2+ymargin]);\n");
    printf("texts=[];\n");
    printf("lines=[];\n");

    // draw the large-healpix boundaries.
    for (hp=0; hp<12; hp++) {
        double xyz[3];

        double crd[6*2];
        double xy[] = { 0.0,0.001,   0.0,1.0,   0.999,1.0,
                        1.0,0.999,   1.0,0.0,   0.001,0.0 };
        for (i=0; i<6; i++) {
            healpix_to_xyzarr(hp, 1, xy[i*2], xy[i*2+1], xyz);
            xyzarr2radec(xyz, crd+i*2+0, crd+i*2+1);
        }
        printf("xy=[");
        for (i=0; i<7; i++)
            printf("%g,%g;", crd[(i%6)*2+0], crd[(i%6)*2+1]);
        printf("];\n");
        printf("[la, lb] = wrapline(xy(:,1),xy(:,2));\n");
        printf("set(la, 'Color', 'b');\n");
        printf("set(lb, 'Color', 'b');\n");
    }

    if (do_neighbours) {
        for (hp=0; hp<HP; hp++) {
            uint neigh[8];
            uint nn;
            nn = healpix_get_neighbours(hp, neigh, Nside);
            for (i=0; i<nn; i++) {
                printf("[la,lb]=wrapline([%g,%g],[%g,%g]);\n",
                       radecs[2*hp], radecs[2*neigh[i]], 
                       radecs[2*hp+1], radecs[2*neigh[i]+1]);
                printf("set([la,lb], "
                       "'Color', [0.5,0.5,0.5], "
                       "'Marker', 'o', "
                       "'MarkerEdgeColor', 'k', "
                       //"'MarkerFaceColor', 'none', "
                       "'MarkerFaceColor', 'white', "
                       "'MarkerSize', %g);\n", markersize);
                printf("set(lb, 'LineStyle', '--');\n");
            }
        }
    }

    for (hp=0; hp<HP; hp++) {
        printf("texts(%i)=text(%g, %g, '",
               hp+1, radecs[2*hp], radecs[2*hp+1]);

        switch (mode) {
        case MODE_XY:
            printf("%i", hp);
            break;
        case MODE_XY_D:
            {
                uint bighp, x, y;
                healpix_decompose_xy(hp, &bighp, &x, &y, Nside);
                printf("%i,%i,%i", bighp, x, y);
            }
            break;
        case MODE_RING:
            printf("%i", healpix_xy_to_ring(hp, Nside));
            break;
        case MODE_RING_D:
            {
                uint ring;
                uint ringnum, longind;
                ring = healpix_xy_to_ring(hp, Nside);
                healpix_decompose_ring(ring, Nside, &ringnum, &longind);
                printf("%i,%i", ringnum, longind);
            }
            break;
        case MODE_NESTED:
            printf("%i", healpix_xy_to_nested(hp, Nside));
            break;
        }

        printf("', 'HorizontalAlignment', 'center', 'FontSize', %g);\n", fontsize);
    }

    // Verify decompose / compose RING.
    for (hp=0; hp<HP; hp++) {
        uint ring, longind;
        int hp2;
        healpix_decompose_ring(hp, Nside, &ring, &longind);
        hp2 = healpix_compose_ring(ring, longind, Nside);
        if (hp2 != hp) {
            fprintf(stderr, "Error: %i -> ring %i, longind %i -> %i.\n",
                    hp, ring, longind, hp2);
        }
    }

    // Verify   XY -> RING -> XY
    for (hp=0; hp<HP; hp++) {
        int ring, hp2;
        ring = healpix_xy_to_ring(hp, Nside);
        hp2 = healpix_ring_to_xy(ring, Nside);
        if (hp2 != hp) {
            uint bighp, x, y;
            uint bighp2, x2, y2;
            uint ringind, longind;
            healpix_decompose_xy(hp, &bighp, &x, &y, Nside);
            healpix_decompose_xy(hp2, &bighp2, &x2, &y2, Nside);
            healpix_decompose_ring(ring, Nside, &ringind, &longind);
            fprintf(stderr, "Error: hp %i (bighp %i, x %i, y %i) -> ring %i (ring %i, long %i) -> hp %i (bighp %i, x %i, y %i).\n",
                    hp, bighp, x, y, ring, ringind, longind, hp2, bighp2, x2, y2);
        }
    }

    free(radecs);

    return 0;
}

