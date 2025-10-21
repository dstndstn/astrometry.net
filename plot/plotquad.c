/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <math.h>
#include <string.h>
#include <stdint.h>

#include <cairo.h>

#include "an-bool.h"
#include "cairoutils.h"
#include "boilerplate.h"
#include "bl.h"
#include "permutedsort.h"
#include "matchfile.h"
#include "log.h"
#include "os-features.h" // for HAVE_NETPBM.

#define OPTIONS "hW:H:w:I:C:PRo:d:cm:s:b:vp"

static void printHelp(char* progname) {
    BOILERPLATE_HELP_HEADER(stdout);
    printf("\nUsage: %s [options] <quads>  > output.png\n"
           "  [-I <input-image>]  Input image (PPM format) to plot over.\n"
           "  [-p]: Input image is PNG format, not PPM.\n"
           "  [-P]              Write PPM output instead of PNG.\n"
           "  [-C <color>]      Color to plot in: (default: white)\n",
           progname);
    cairoutils_print_color_names("\n                 ");
    printf("\n"
	   "  [-b <color>]      Draw in <color> behind each line.\n"
           "  [-c]:            Also plot a circle at each vertex.\n"
           "  [-W <width> ]       Width of output image.\n"
           "  [-H <height>]       Height of output image.\n"
           "  [-w <width>]      Width of lines to draw (default: 5).\n"
           "  [-R]:  Read quads from stdin.\n"
           "  [-o <opacity>]\n"
           "  [-d <dimension of \"quad\">]\n"
           "  [-s <scale>]: scale quad coordinates before plotting.\n"
           "\n"
           " ( [-m <match.fits>]: Get quad from match file.\n"
           " OR  <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> [...])\n"
           "\n");
}

int main(int argc, char *args[]) {
    int argchar;
    char* progname = args[0];
    int W = 0, H = 0;
    int lw = 5;
    int nquads;
    int i;
    dl* coords;
    char* infn = NULL;
    anbool pngoutput = TRUE;
    anbool pnginput = FALSE;
    anbool fromstdin = FALSE;
    anbool randomcolor = FALSE;
    float a = 1.0;
    anbool plotmarker = FALSE;

    unsigned char* img = NULL;
    cairo_t* cairo;
    cairo_surface_t* target;
    float r=1.0, g=1.0, b=1.0;
    int dimquads = 4;
    double scale = 1.0;
    char* matchfn = NULL;

    anbool background = FALSE;
    float br=0.0, bg=0.0, bb=0.0;

    int loglvl = LOG_MSG;

    coords = dl_new(16);

    while ((argchar = getopt(argc, args, OPTIONS)) != -1)
        switch (argchar) {
        case 'v':
            loglvl++;
            break;
        case 's':
            scale = atof(optarg);
            break;
        case 'm':
            matchfn = optarg;
            break;
        case 'c':
            plotmarker = TRUE;
            break;
        case 'd':
            dimquads = atoi(optarg);
            break;
        case 'C':
            if (!strcasecmp(optarg, "random"))
                randomcolor = TRUE;
            else if (cairoutils_parse_color(optarg, &r, &g, &b)) {
                fprintf(stderr, "I didn't understand color \"%s\".\n", optarg);
                exit(-1);
            }
            break;
        case 'b':
            background = TRUE;
            if (cairoutils_parse_color(optarg, &br, &bg, &bb)) {
                fprintf(stderr, "I didn't understand color \"%s\".\n", optarg);
                exit(-1);
            }
            break;
        case 'o':
            a = atof(optarg);
            break;
        case 'R':
            fromstdin = TRUE;
            break;
        case 'I':
            infn = optarg;
            break;
        case 'P':
            pngoutput = FALSE;
            break;
        case 'p':
            pnginput = TRUE;
            break;
        case 'W':
            W = atoi(optarg);
            break;
        case 'H':
            H = atoi(optarg);
            break;
        case 'w':
            lw = atoi(optarg);
            break;
        case 'h':
        case '?':
        default:
            printHelp(progname);
            exit(-1);
        }

    log_init(loglvl);
    log_to(stderr);

    if (dimquads == 0) {
        printf("Error: dimquads (-d) must be positive.\n");
        printHelp(progname);
        exit(-1);
    }

    if (!fromstdin && ((argc - optind) % (2*dimquads))) {
        printHelp(progname);
	printf("With quads of dimension %i (-d), expected %i command-line args, but got %i\n", dimquads, 2*dimquads, argc-optind);
        exit(-1);
    }
    if (!((W && H) || infn)) {
        printHelp(progname);
	printf("Need either width and height, or input filename\n");
        exit(-1);
    }
    if (infn && (W || H)) {
        printf("Error: if you specify an input file, you can't give -W or -H (width or height) arguments.\n\n");
        printHelp(progname);
        exit(-1);
    }

    if (matchfn) {
        matchfile* mf = matchfile_open(matchfn);
        MatchObj* mo;
        if (!mf) {
            fprintf(stderr, "Failed to open matchfile \"%s\".\n", matchfn);
            exit(-1);
        }
        while (1) {
            mo = matchfile_read_match(mf);
            if (!mo)
                break;
            for (i=0; i<2*dimquads; i++) {
                dl_append(coords, mo->quadpix[i]);
            }
        }
    }
    for (i=optind; i<argc; i++) {
        double pos = atof(args[i]);
        dl_append(coords, pos);
    }
    if (fromstdin) {
        for (;;) {
            int j;
            double p;
            if (feof(stdin))
                break;
            for (j=0; j<(2*dimquads); j++) {
                if (fscanf(stdin, " %lg", &p) != 1) {
                    fprintf(stderr, "Failed to read a quad from stdin.\n");
                    exit(-1);
                }
                dl_append(coords, p);
            }
        }
    }

    if (scale != 1.0) {
        for (i=0; i<dl_size(coords); i++) {
            dl_set(coords, i, scale * dl_get(coords, i));
        }
    }

    nquads = dl_size(coords) / (2*dimquads);

    if (infn) {
#if HAVE_NETPBM
#else
        logverb("No netpbm available: forcing PNG input.\n");
        pnginput = TRUE;
#endif
        if (pnginput) {
            logverb("Reading PNG file %s\n", infn);
            img = cairoutils_read_png(infn, &W, &H);
        }
#if HAVE_NETPBM
        else {
            logverb("Reading PPM from %s\n", infn);
            cairoutils_fake_ppm_init();
            img = cairoutils_read_ppm(infn, &W, &H);
        }
#else
#endif
        if (!img) {
            fprintf(stderr, "Failed to read input image %s.\n", infn);
            exit(-1);
        }
        cairoutils_rgba_to_argb32(img, W, H);
    } else {
        // Allocate a black image.
        img = calloc(4 * W * H, 1);
    }

    target = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, W, H, W*4);
    cairo = cairo_create(target);
    cairo_set_line_width(cairo, lw);
    cairo_set_line_join(cairo, CAIRO_LINE_JOIN_BEVEL);
    //cairo_set_line_join(cairo, CAIRO_LINE_JOIN_ROUND);
    cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);

    if (!randomcolor)
        cairo_set_source_rgba(cairo, r, g, b, a);

    for (i=0; i<nquads; i++) {
        int j;
        double theta[dimquads];
        int perm[dimquads];
        double cx, cy;

        // Make the quad convex so Sam's eyes don't bleed.
        cx = cy = 0.0;
        for (j=0; j<dimquads; j++) {
            cx += dl_get(coords, i*(2*dimquads) + j*2);
            cy += dl_get(coords, i*(2*dimquads) + j*2 + 1);
        }
        cx /= dimquads;
        cy /= dimquads;
        for (j=0; j<dimquads; j++) {
            theta[j] = atan2(dl_get(coords, i*(2*dimquads) + j*2 + 1)-cy,
                             dl_get(coords, i*(2*dimquads) + j*2 + 0)-cx);
        }
        permutation_init(perm, dimquads);
        permuted_sort(theta, sizeof(double), compare_doubles_asc, perm, dimquads);

        // hack.
        if (background) {
            cairo_save(cairo);
            cairo_set_line_width(cairo, lw + 2.0);
            cairo_set_source_rgba(cairo, br, bg, bb, 0.75);
            for (j=0; j<dimquads; j++) {
                ((j==0) ? cairo_move_to : cairo_line_to)
                    (cairo,
                     dl_get(coords, i*(2*dimquads) + perm[j]*2),
                     dl_get(coords, i*(2*dimquads) + perm[j]*2 + 1));
            }
            cairo_close_path(cairo);
            cairo_stroke(cairo);
            cairo_restore(cairo);
        }

        if (randomcolor) {
            r = ((rand() % 128) + 127) / 255.0;
            g = ((rand() % 128) + 127) / 255.0;
            b = ((rand() % 128) + 127) / 255.0;
            cairo_set_source_rgba(cairo, r, g, b, a);
        }
        for (j=0; j<dimquads; j++) {
            ((j==0) ? cairo_move_to : cairo_line_to)
                (cairo,
                 dl_get(coords, i*(2*dimquads) + perm[j]*2),
                 dl_get(coords, i*(2*dimquads) + perm[j]*2 + 1));
        }
        cairo_close_path(cairo);
        cairo_stroke(cairo);
    }

    if (plotmarker) {
        for (i=0; i<dl_size(coords)/2; i++) {
            double x = dl_get(coords, i*2 + 0);
            double y = dl_get(coords, i*2 + 1);
            double rad = 5;
            cairo_arc(cairo, x, y, rad, 0.0, 2.0*M_PI);
            cairo_stroke(cairo);
        }
    }

    // Convert image for output...
    cairoutils_argb32_to_rgba(img, W, H);

    if (pngoutput) {
        if (cairoutils_stream_png(stdout, img, W, H)) {
            fprintf(stderr, "Failed to write PNG.\n");
            exit(-1);
        }
    } else {
        if (cairoutils_stream_ppm(stdout, img, W, H)) {
            fprintf(stderr, "Failed to write PPM.\n");
            exit(-1);
        }
    }

    cairo_surface_destroy(target);
    cairo_destroy(cairo);
    free(img);

    return 0;
}
