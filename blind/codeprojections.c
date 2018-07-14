/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/**
 Reads a .code or .ckdt file, projects each code onto each pair of axes,
 and histograms the results.  Writes out the histograms as Matlab literals.

 Pipe the output to a file like "hists.m", then in Matlab run the
 "codeprojections.m" script.

 HACK - I haven't looked at how code dimensionality (dimcodes)
 influences the "volume_at_value()" function.  The volume-corrected plots
 may therefore be wrong.
 */

#include <string.h>
#include <limits.h>
#include <math.h>

#include "starutil.h"
#include "codekd.h"
#include "kdtree_fits_io.h"
#include "keywords.h"
#include "boilerplate.h"

#define OPTIONS "hd"


static void print_help(char* progname)
{
    BOILERPLATE_HELP_HEADER(stderr);
    fprintf(stderr, "Usage: %s  <code kdtree>\n"
            "       [-d]: normalize by volume (produce density plots)\n\n",
            progname);
}

// 2-D hists
int** hists = NULL;
double** dhists = NULL;
int Nbins = 40;
int Dims;

// 2-D hist of {C,D}x,{C,D}y
int* xyhist = NULL;
double* dxyhist = NULL;

// 1-D hists
int* single = NULL;
double* dsingle = NULL;
int Nsingle = 100;

anbool do_density = FALSE;

double minvalue;
double scale;

static Const double volume_at_value(double x) {
    // codes in a circle live inside the circle
    //    (x-1/2)^2 + (y-1/2)^2 = 1/2
    // we are given "x" and want to find the distance
    // between the upper and lower arcs of the circle;
    // ie y(x)_upper - y(x)_lower.  Hence we don't care
    // about the y offset of the center of the circle and
    // we want twice the value y(x)_upper.  Ie, solve
    //    (x-1/2)^2 + y^2 = 1/2
    // for y, and return twice that.
    //    y = sqrt(1/2 - (x - 1/2)^2).
    //      = sqrt(1/2 - (x^2 - x + 1/4)
    //      = sqrt(-x^2 + x + 1/4)
    return 2.0 * sqrt(-x*x + x + 0.25);
}

static int value_to_bin(double val, int Nbins) {
    int bin = (int)((val - minvalue) * scale * Nbins);
    if (bin >= Nbins) {
        bin = Nbins-1;
        printf("truncating value %g\n", val);
    }
    if (bin < 0) {
        bin = 0;
        printf("truncating (up) value %g\n", val);
    }
    return bin;
}

static void add_to_single_histogram(int dim, double val) {
    int* hist = single + Nsingle * dim;
    int bin = value_to_bin(val, Nsingle);
    hist[bin]++;
    if (do_density) {
        double* dhist = dsingle + Nsingle * dim;
        dhist[bin] += 1.0 / volume_at_value(val);
    }
}

static void add_to_histogram(int dim1, int dim2, double val1, double val2) {
    int xbin, ybin;
    int* hist = hists[dim1 * Dims + dim2];
    xbin = value_to_bin(val1, Nbins);
    ybin = value_to_bin(val2, Nbins);
    hist[xbin * Nbins + ybin]++;
    if (do_density) {
        double* dhist = dhists[dim1 * Dims + dim2];
        double inc;
        if (dim1/2 == dim2/2)
            // (cx vs cy) or (dx vs dy); the other two dimensions are independent.
            inc = 1.0;
        else
            inc = 1.0 / (volume_at_value(val1) * volume_at_value(val2));
        dhist[xbin * Nbins + ybin] += inc;
    }
}

static void add_to_cd_histogram(double val1, double val2) {
    int xbin, ybin;
    xbin = value_to_bin(val1, Nbins);
    ybin = value_to_bin(val2, Nbins);
    xyhist[xbin * Nbins + ybin]++;
    if (do_density)
        dxyhist[xbin * Nbins + ybin] += 1.0 / (volume_at_value(val1) * volume_at_value(val2));
}

int main(int argc, char *argv[])
{
    int argchar;
    char *ckdtfname = NULL;
    int i, j, d, e;
    anbool circle;
    codetree* ct = NULL;
    kdtree_t* ckdt = NULL;
    int Ncodes;
    int dimcodes;

    if (argc <= 2) {
        print_help(argv[0]);
        return (OPT_ERR);
    }

    while ((argchar = getopt (argc, argv, OPTIONS)) != -1)
        switch (argchar) {
        case 'd':
            do_density = TRUE;
            break;
        case 'h':
            print_help(argv[0]);
            return (HELP_ERR);
        default:
            return (OPT_ERR);
        }

    if (optind != argc-1) {
        print_help(argv[0]);
        printf("You must give a code kdtree filename!\n");
        exit(-1);
    }
    ckdtfname = argv[optind];

    ct = codetree_open(ckdtfname);
    if (!ct) {
        fprintf(stderr, "Failed to read code kdtree file %s.\n", ckdtfname);
        exit(-1);
    }
    circle = qfits_header_getboolean(ct->header, "CIRCLE", 0);
    ckdt = ct->tree;
    Ncodes = ckdt->ndata;
    dimcodes = ckdt->ndim;

    fprintf(stderr, "Index %s the CIRCLE property.\n",
            (circle ? "has" : "does not have"));

    if (circle) {
        double margin = 0.1;
        minvalue = 0.5 - M_SQRT1_2 - (0.5 * margin);
        //scale = M_SQRT1_2 + margin;
        scale = 1.0 / (M_SQRT2 + margin);
    } else {
        double margin = 0.06;
        minvalue = 0.0 - (0.5 * margin);
        scale = 1.0 / (1.0 + margin);

        if (do_density) {
            fprintf(stderr, "Warning: this index does not have the CIRCLE property "
                    "so the -d flag has no effect.\n");
            do_density = FALSE;
        }
    }

    // Allocate memory for projection histograms
    hists  = calloc(dimcodes * dimcodes, sizeof(int*));
    dhists = calloc(dimcodes * dimcodes, sizeof(double*));

    for (d = 0; d < dimcodes; d++) {
        for (e = 0; e < d; e++) {
            hists [d*dimcodes + e] = calloc(Nbins * Nbins, sizeof(int));
            dhists[d*dimcodes + e] = calloc(Nbins * Nbins, sizeof(double));
        }
        // Since the 4x4 matrix of histograms is actually symmetric,
        // only make half
        for (; e < dimcodes; e++) {
            hists [d*dimcodes + e] = NULL;
            dhists[d*dimcodes + e] = NULL;
        }
    }

    xyhist  = calloc(Nbins * Nbins, sizeof(int));
    dxyhist = calloc(Nbins * Nbins, sizeof(double));

    single  = calloc(dimcodes * Nsingle, sizeof(int));
    dsingle = calloc(dimcodes * Nsingle, sizeof(double));

    for (i=0; i<Ncodes; i++) {
        double code[dimcodes];

        codetree_get(ct, i, code);

        for (d = 0; d < dimcodes; d++) {
            for (e = 0; e < d; e++)
                add_to_histogram(d, e, code[d], code[e]);
            add_to_single_histogram(d, code[d]);
        }
        for (d=0; d<dimcodes/2; d++)
            add_to_cd_histogram(code[2*d], code[2*d+1]);
    }

    codetree_close(ct);

    for (d = 0; d < dimcodes; d++) {
        for (e = 0; e < d; e++) {
            int* hist;
            printf("hist_%i_%i=zeros([%i,%i]);\n",
                   d, e, Nbins, Nbins);
            hist = hists[d * dimcodes + e];
            for (i = 0; i < Nbins; i++) {
                int j;
                printf("hist_%i_%i(%i,:)=[", d, e, i + 1);
                for (j = 0; j < Nbins; j++) {
                    printf("%i,", hist[i*Nbins + j]);
                }
                printf("];\n");
            }
            free(hist);
            if (do_density) {
                double* dhist;
                printf("dhist_%i_%i=zeros([%i,%i]);\n",
                       d, e, Nbins, Nbins);
                dhist = dhists[d * dimcodes + e];
                for (i = 0; i < Nbins; i++) {
                    printf("dhist_%i_%i(%i,:)=[", d, e, i + 1);
                    for (j = 0; j < Nbins; j++)
                        printf("%g,", dhist[i*Nbins + j]);
                    printf("];\n");
                }
                free(dhist);
            }
        }
        printf("hist_%i=[", d);
        for (i = 0; i < Nsingle; i++)
            printf("%i,", single[d*Nsingle + i]);
        printf("];\n");

        if (do_density) {
            printf("dhist_%i=[", d);
            for (i = 0; i < Nsingle; i++)
                printf("%g,", dsingle[d*Nsingle + i]);
            printf("];\n");
        }
    }
    printf("hist_xy=[");
    for (i=0; i<Nbins; i++) {
        for (j=0; j<Nbins; j++)
            printf("%i,", xyhist[i*Nbins+j]);
        printf(";");
    }
    printf("];\n");
    if (do_density) {
        printf("dhist_xy=[");
        for (i=0; i<Nbins; i++) {
            for (j=0; j<Nbins; j++)
                printf("%g,", dxyhist[i*Nbins+j]);
            printf(";");
        }
        printf("];\n");
    }

    free(xyhist);
    free(hists);
    free(single);
    if (do_density) {
        free(dxyhist);
        free(dhists);
        free(dsingle);
    }

    fprintf(stderr, "Done!\n");

    return 0;
}



