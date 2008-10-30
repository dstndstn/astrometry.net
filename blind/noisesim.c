/*
  This file is part of the Astrometry.net suite.
  Copyright 2006, 2007 Dustin Lang, Keir Mierle and Sam Roweis.

  The Astrometry.net suite is free software; you can redistribute
  it and/or modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation, version 2.

  The Astrometry.net suite is distributed in the hope that it will be
  useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the Astrometry.net suite ; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
*/

/**
   This program simulates noise in index star positions and
   computes the resulting noise in the code values that are
   produced.

   See the wiki page:
   -   http://trac.astrometry.net/wiki/ErrorAnalysis
 */
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

#include "starutil.h"
#include "mathutil.h"
#include "noise.h"

const char* OPTIONS = "he:n:da:ri:s:";

void print_help(char* progname) {
    printf("Usage: %s\n"
           "   [-i <index-jitter>]: noise in the index, in arcseconds\n"
           "   [-e <field-jitter>]: noise in the image, in pixels\n"
           "   [-q <dimquads>]: set number of stars per \"quad\"\n"
           "   [-n <n-samples>]: number of samples to draw\n"
           "   [-a <quad-scale>]: angle between stars A and B, in arcminutes\n"
           "   [-s <pixel-scale>]: set the image pixel scale (arcseconds/pixel)\n"
           "   [-d]: print the individual code distances\n"
           "   [-r]: print the index and field code values\n"
           "\n", progname);
}

int main(int argc, char** args) {
	int argchar;

	double ABangle = 4.0; // arcminutes
	double pixscale = 0.396; // arcsec/pixel
	//	double pixscale = 0.05; // Hubble
	double pixnoise = 1; // pixels, stddev
	double indexjitter = 1.0; // arcsec, stddev
	double noisedist;

    int dimquads = 4;
    int dimcodes;

	double realxyz[3 * DQMAX];
    double indexxyz[3 * DQMAX];

	double ra, dec;
	int j;
	int k;
	int N=1000;

	int printdists = FALSE;

	dl* codedelta;
	dl* codedists;
	dl* noises;

    int printcodes = FALSE;

    double* realA;
    double* realB;

    /*
     int abInvalid = 0;
     int cdInvalid = 0;
     */

	noises = dl_new(16);

    srand((unsigned int)time(NULL));

	while ((argchar = getopt (argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case '?':
            print_help(args[0]);
            exit(-1);
        case 'q':
            dimquads = atoi(optarg);
            break;
        case 'i':
            indexjitter = atof(optarg);
            break;
		case 'e':
			pixnoise = atof(optarg);
			dl_append(noises, pixnoise);
			break;
		case 'd':
			printdists = TRUE;
			break;
		case 'n':
			N = atoi(optarg);
			break;
		case 'a':
			ABangle = atof(optarg);
			break;
        case 'r':
            printcodes = TRUE;
            break;
        case 's':
            pixscale = atof(optarg);
            break;
		}

    if (optind != argc-1) {
        print_help(args[0]);
        printf("Unknown extra args.\n");
        exit(-1);
    }

    if (dimquads > DQMAX || dimquads < 3) {
        fprintf(stderr, "Invalid dimquads: must be in [3, %i]\n", DQMAX);
        exit(-1);
    }
	dimcodes = dimquad2dimcode(dimquads);

    realA = realxyz;
    realB = realxyz + 3;

	// A
	ra = 0.0;
	dec = 0.0;
	radec2xyzarr(ra, dec, realA);

	// B
	ra = arcmin2rad(ABangle);
	dec = 0.0;
	radec2xyzarr(ra, dec, realB);

	if (printcodes) {
		printf("icode=[];\n");
		printf("fcode=[];\n");
	}

    printf("pixscale=%g;\n", pixscale);
 	printf("quadsize=%g;\n", ABangle);
	printf("codemean=[];\n");
	printf("codestd=[];\n");
	printf("pixerrs=[];\n");
    printf("indexnoise=%g;\n", indexjitter);

    /*
     printf("abinvalid=[];\n");
     printf("cdinvalid=[];\n");
     printf("scale=[];\n");
     */

	if (dl_size(noises) == 0)
		dl_append(noises, pixnoise);

	for (k=0; k<dl_size(noises); k++) {
        double mean, std;

		pixnoise = dl_get(noises, k);
		noisedist = arcsec2dist(indexjitter);

		codedelta = dl_new(256);
		codedists = dl_new(256);

		//abInvalid = cdInvalid = 0;

		for (j=0; j<N; j++) {
			double midAB[3];
			double fcode[dimcodes];
			double icode[dimcodes];
			double field[dimquads * 2];
			int i;
            bool ok = TRUE;
            double scale;

			star_midpoint(midAB, realA, realB);

			// place interior stars uniformly in the circle around the midpoint of AB.
            for (i=2; i<dimquads; i++)
                sample_star_in_circle(midAB, ABangle/2.0, realxyz + 3*i);

			// add noise to real star positions to yield index positions
            for (i=0; i<dimquads; i++)
                add_star_noise(realxyz + 3*i, noisedist, indexxyz + 3*i);

			compute_star_code(indexxyz, dimquads, icode);

			// project to field coords
            for (i=0; i<dimquads; i++)
                ok &= star_coords(realxyz + 3*i, midAB, field + 2*i, NULL);
            assert(ok);
			// scale to pixels.
			for (i=0; i<(dimquads*2); i++)
				field[i] = rad2arcsec(field[i]) / pixscale;

			// add field noise to get image coordinates
            for (i=0; i<dimquads; i++)
                add_field_noise(field + 2*i, pixnoise, field + 2*i);

			compute_field_code(field, dimquads, fcode, &scale);

            /*
             if ((scale < square(lowerAngle * 60.0 / pixscale)) ||
             (scale > square(upperAngle * 60.0 / pixscale)))
             abInvalid++;
             else if ((((codecx*codecx - codecx) + (codecy*codecy - codecy)) > 0.0) ||
             (((codedx*codedx - codedx) + (codedy*codedy - codedy)) > 0.0))
             cdInvalid++;
             if (matlab)
             printf("scale(%i)=%g;\n", j+1, sqrt(scale)*pixscale/60.0);
             */

            if (printcodes) {
                printf("icode(%i,:)=[", j+1);
                for (i=0; i<dimcodes; i++)
                    printf("%g,", icode[i]);
                printf("];\n");
                printf("fcode(%i,:)=[", j+1);
                for (i=0; i<dimcodes; i++)
                    printf("%g,", fcode[i]);
                printf("];\n");
            }

			dl_append(codedists, sqrt(distsq(icode, fcode, dimcodes)));

            /*
             compute the true and noisy transforms, sample stars some number
             of quad radiuses away from the quad center, transform them
             through the two transforms and measure the distance between the
             transformed positions...
             */

		}

        if (printdists) {
            printf("codedists%i=[", k+1);
            for (j=0; j<dl_size(codedists); j++)
                printf("%g,", dl_get(codedists, j));
            printf("];\n");
        }

        mean = 0.0;
        for (j=0; j<dl_size(codedists); j++)
            mean += dl_get(codedists, j);
        mean /= (double)dl_size(codedists);
        std = 0.0;
        for (j=0; j<dl_size(codedists); j++)
            std += square(dl_get(codedists, j) - mean);
        std /= ((double)dl_size(codedists) - 1);
        std = sqrt(std);

		printf("pixerrs(%i)=%g;\n", k+1, pixnoise);
        printf("codemean(%i)=%g;\n", k+1, mean);
        printf("codestd(%i)=%g;\n", k+1, std);

        /*
         printf("abinvalid(%i) = %g;\n", k+1, abInvalid / (double)N);
         printf("cdinvalid(%i) = %g;\n", k+1, cdInvalid / (double)N);
         */

		dl_free(codedelta);
		dl_free(codedists);
	}

	dl_free(noises);
	
	return 0;
}
