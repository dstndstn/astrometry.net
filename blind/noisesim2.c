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
 This program simulates noise in the matched quad to see how much error this
 generates for other stars in the image.

 It works purely in the image pixel coordinate space.
 **/
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "starutil.h"
#include "mathutil.h"
#include "svd.h"
#include "noise.h"

const char* OPTIONS = "hi:f:q:n:a:s:LQ:";

void print_help(char* progname) {
    printf("Usage: %s\n"
           "   [-i <index-jitter>]: noise in the index stars, in pixels; default 1\n"
           "   [-f <field-jitter>]: noise in the image, in pixels; default 1\n"
           "   [-Q <index quad jitter>]: noise to add to the index quad stars; default = same as the rest of the index stars.\n"
           "   [-q <dimquads>]: set number of stars per \"quad\"; default 4\n"
           "   [-n <n-samples>]: number of stars to draw; default 1000\n"
           "   [-a <quad-scale>]: distance between stars A and B, in pixels; default 100\n"
           "   [-s <image-size>]: edge size of the image; default 1000\n"
           "   [-L]: sample stars uniformly in a line\n"
           "\n", progname);
}

struct transform {
    double scale;
    double rotation[4];
    double incenter[2];
    double outcenter[2];
};
typedef struct transform transform;

void transform_print(const transform* T) {
    printf("  in offset [ %g, %g ]\n", T->incenter[0], T->incenter[1]);
    printf("  scale %g\n", T->scale);
    printf("  rotation [ %g, %g ]\n", T->rotation[0], T->rotation[1]);
    printf("           [ %g, %g ]\n", T->rotation[2], T->rotation[3]);
    printf("  out offset [ %g, %g ]\n", T->outcenter[0], T->outcenter[1]);
}

void apply_transform(const double* in, const transform* t, double* out) {
    double p[2], rp[2];
    int i, j;

    for (i=0; i<2; i++) {
        p[i] = in[i] - t->incenter[i];
    }

    for (i=0; i<2; i++) {
        p[i] *= t->scale;
    }

    for (i=0; i<2; i++) {
        rp[i] = 0;
        for (j=0; j<2; j++) {
            rp[i] += p[j] * t->rotation[2*i + j];
            //rp[i] += p[j] * t->rotation[2*j + i];
        }
    }

    for (i=0; i<2; i++) {
        out[i] = rp[i] + t->outcenter[i];
    }
}

void procrustes(const double* index,
                const double* field,
                int N,
                transform* t) {
	int i, j, k;
	double index_cm[2] = {0, 0};
	double field_cm[2] = {0, 0};
	double cov[4] = {0, 0, 0, 0};
	double U[4], V[4], S[2];
	double R[4] = {0, 0, 0, 0};
	double scale;
    double ivar=0, fvar=0;

    // get centers of mass of the corresponding points.
	for (i=0; i<N; i++) {
        for (j=0; j<2; j++) {
            index_cm[j] += index[2*i+j] / (double)N;
            field_cm[j] += field[2*i+j] / (double)N;
        }
    }

	// compute the covariance
	for (i=0; i<N; i++)
		for (j=0; j<2; j++)
			for (k=0; k<2; k++)
				//cov[j*2 + k] += (index[i*2 + k] - index_cm[k]) * (field[i*2 + j] - field_cm[j]);
				cov[j*2 + k] += (index[i*2 + j] - index_cm[j]) * (field[i*2 + k] - field_cm[k]);

	// -run SVD
	{
		double* pcov[] = { cov, cov+2 };
		double* pU[]   = { U,   U  +2 };
		double* pV[]   = { V,   V  +2 };
		double eps, tol;
		eps = 1e-30;
		tol = 1e-30;
		svd(2, 2, 1, 1, eps, tol, pcov, S, pU, pV);
	}

	// -compute rotation matrix R = V U'
	for (i=0; i<2; i++)
		for (j=0; j<2; j++)
			for (k=0; k<2; k++)
				R[i*2 + j] += V[i*2 + k] * U[j*2 + k];

	// -compute scale: make the variances equal.
    for (i=0; i<N; i++)
        for (j=0; j<2; j++) {
            ivar += square(index[2*i + j] - index_cm[j]);
            fvar += square(field[2*i + j] - field_cm[j]);
        }
    scale = sqrt(fvar / ivar);

    t->scale = scale;
    memcpy(t->rotation, R, 4*sizeof(double));
    for (i=0; i<2; i++) {
        t->incenter [i] = index_cm[i];
        t->outcenter[i] = field_cm[i];
    }
}



int main(int argc, char** args) {
	int argchar;

	double abdist = 100.0; // pixels
	double fieldnoise = 1.0; // pixels, stddev
	double indexnoise = 1.0; // pixels, stddev
    double imgsize = 1000.0; // pixels

    int dimquads = 4;
	int N=1000;
    int dimcodes;

    int i;

    double cx, cy;
    double center[2];

    double fieldcode[2 * DCMAX];
    double indexcode[2 * DCMAX];
    double fieldscale;
    double indexscale;
    double codedist;

    double* fieldstars;
    double* indexstars;
    double* indexproj;

    double indexquadnoise = -1.0;

    int line = 0;

    transform T;

    srand((unsigned int)time(NULL));
    //srand(0);

	while ((argchar = getopt (argc, args, OPTIONS)) != -1)
		switch (argchar) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case '?':
            print_help(args[0]);
            exit(-1);
        case 'Q':
            indexquadnoise = atof(optarg);
        case 'L':
            line = 1;
            break;
        case 'q':
            dimquads = atoi(optarg);
            break;
        case 'i':
            indexnoise = atof(optarg);
            break;
		case 'f':
			fieldnoise = atof(optarg);
			break;
		case 'n':
			N = atoi(optarg);
			break;
		case 'a':
			abdist = atof(optarg);
			break;
        case 's':
            imgsize = atof(optarg);
            break;
		}

    if (optind != argc) {
        print_help(args[0]);
        printf("Unknown extra args (%i / %i).\n", optind, argc);
        exit(-1);
    }

    if (dimquads > DQMAX || dimquads < 3) {
        fprintf(stderr, "Invalid dimquads: must be in [3, %i]\n", DQMAX);
        exit(-1);
    }
	dimcodes = dimquad2dimcode(dimquads);

    if (indexquadnoise == -1.0) {
        indexquadnoise = indexnoise;
    }

    fieldstars = malloc(2 * N * sizeof(double));
    indexstars = malloc(2 * N * sizeof(double));
    indexproj = malloc(2 * N * sizeof(double));


    cx = imgsize / 2.0;
    cy = imgsize / 2.0;
    center[0] = cx;
    center[1] = cy;

    // A
    fieldstars[0] = cx - abdist / 2.0;
    fieldstars[1] = cy;
    // B
    fieldstars[2] = cx + abdist / 2.0;
    fieldstars[3] = cy;

    // CDE...
    for (i=2; i<dimquads; i++) {
        sample_in_circle(center, abdist/2.0, fieldstars + 2*i);
    }

    // the rest of the stars in the field...
    if (line) {
        for (i=dimquads; i<N; i++) {
            /* Sample with uniform density in L^2 space:
             double s = uniform_sample(-0.5, 0.5);
             s = sqrt(fabs(s)) * (s > 0 ? 1 : -1);
             fieldstars[2*i + 0] = (s+0.5) * imgsize;
             */
            fieldstars[2*i + 0] = uniform_sample(0.0, imgsize);
            fieldstars[2*i + 1] = cy;
        }
    } else {
        for (i=dimquads; i<N; i++) {
            fieldstars[2*i + 0] = uniform_sample(0.0, imgsize);
            fieldstars[2*i + 1] = uniform_sample(0.0, imgsize);
        }
    }

    memcpy(indexstars, fieldstars, 2*N*sizeof(double));


    // add noise to index stars...
    for (i=0; i<dimquads; i++) {
        add_field_noise(indexstars + 2*i, indexquadnoise, indexstars + 2*i);
    }
    for (i=dimquads; i<N; i++) {
        add_field_noise(indexstars + 2*i, indexnoise, indexstars + 2*i);
    }

    // add noise to field stars...
    for (i=0; i<N; i++) {
        add_field_noise(fieldstars + 2*i, fieldnoise, fieldstars + 2*i);
    }

    compute_field_code(fieldstars, dimquads, fieldcode, &fieldscale);
    compute_field_code(indexstars, dimquads, indexcode, &indexscale);

    codedist = sqrt(distsq(fieldcode, indexcode, dimcodes));

    // compute rigid Procrustes transformation from index to field coords.

    printf("Before: d2 is %g\n", distsq(fieldstars, indexstars, 2*dimquads));

    procrustes(indexstars, fieldstars, dimquads, &T);

    printf("Transform is:\n");
    transform_print(&T);

    /*{
     double idist = 0;
     double fdist = 0;
     int j;
     for (i=0; i<dimquads; i++) {
     for (j=0; j<2; j++) {
     idist += square(indexstars[2*i + j] - T.incenter[j]);
     }
     }
     for (i=0; i<dimquads; i++) {
     for (j=0; j<2; j++) {
     fdist += square(fieldstars[2*i + j] - T.outcenter[j]);
     }
     }
     printf("idist %g, fdist %g, scale %g\n", idist, fdist, T.scale);
     }
     */

    // project index stars through the transform.
    for (i=0; i<N; i++) {
        apply_transform(indexstars + 2*i, &T, indexproj + 2*i);
    }

    printf("After: d2 is %g\n", distsq(fieldstars, indexproj, 2*dimquads));

    /*
     fprintf(stderr, "indexstars=[");
     for (i=0; i<N; i++)
     fprintf(stderr, "%g,%g;", indexstars[2*i], indexstars[2*i+1]);
     fprintf(stderr, "];\n");
     */

    fprintf(stderr, "codedist=%g;\n", codedist);

    fprintf(stderr, "fieldstars=[");
    for (i=0; i<N; i++)
        fprintf(stderr, "%g,%g;", fieldstars[2*i], fieldstars[2*i+1]);
    fprintf(stderr, "];\n");

    fprintf(stderr, "indexproj=[");
    for (i=0; i<N; i++)
        fprintf(stderr, "%g,%g;", indexproj[2*i], indexproj[2*i+1]);
    fprintf(stderr, "];\n");

    {
        double fdist = 0;
        int j;
        for (i=0; i<dimquads; i++) {
            for (j=0; j<2; j++) {
                fdist += square(fieldstars[2*i + j] - T.outcenter[j]);
            }
        }
        fdist /= (double)dimquads;
        fdist = sqrt(fdist);
        fprintf(stderr, "H=%g;\n", fdist);
    }

    fprintf(stderr, "data=[");

    for (i=0; i<N; i++) {
        double L, E;
        L = sqrt(distsq(fieldstars + 2*i, T.outcenter, 2));
        E = sqrt(distsq(fieldstars + 2*i, indexproj + 2*i, 2));

        fprintf(stderr, "%g, %g,", L, E);

        fprintf(stderr, ";");
    }

    fprintf(stderr, "];\n");

    fprintf(stderr,
            "L = data(:,1);\n"
            "E = data(:,2);\n"
            );

	return 0;
}
