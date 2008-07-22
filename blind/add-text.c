#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <sys/param.h>

#include "an-bool.h"

#include <cairo.h>
#include "cairoutils.h"
#include "bl.h"

const char* OPTIONS = "hj:t:f:c:x:y:W:H:o:";

void print_help(char* progname) {
    int i;
    fprintf(stderr, "\nUsage: %s\n"
           "  -j <jpeg-filename>\n"
           "  -o <output-jpeg-filename>\n"
           "  -t <text>\n"
           "  -f <font>\n"
           "  -c <color>:\n"
           , progname);
    for (i=0;; i++) {
        const char* color = cairoutils_get_color_name(i);
        if (!color) break;
        fprintf(stderr, "     %s\n", color);
    }
    fprintf(stderr, "  -x <x-center>\n"
            "  -y <y-center>\n"
            "  -W <width>\n"
            "  -H <height>\n");
}

extern char *optarg;
extern int optind, opterr, optopt;

int main(int argc, char** args) {
    int c;
    char* jpegfn = NULL;
    char* outfn = NULL;
    cairo_t* cairo = NULL;
    cairo_surface_t* target = NULL;
    double fontsize = 50.0;
    int imW, imH;
    double xc=-1, yc=-1, W=0, H=0;
    unsigned char* img = NULL;
    char* text = NULL;
    char* font = "Purisa";
    float r=1.0, g=1.0, b=1.0;
    sl* lines;
    char* cptr;
    int i;
    cairo_text_extents_t textents;
    double txtw, txth;
    double y;
    int res;
    double linespacing = 0.2;

    while ((c = getopt(argc, args, OPTIONS)) != -1) {
        switch (c) {
        case 'h':
            print_help(args[0]);
            exit(0);
        case 'j':
            jpegfn = optarg;
            break;
        case 'o':
            outfn = optarg;
            break;
        case 't':
            text = optarg;
            break;
        case 'f':
            font = optarg;
            break;
        case 'c':
            if (cairoutils_parse_color(optarg, &r, &g, &b)) {
                fprintf(stderr, "I didn't understand color \"%s\".\n", optarg);
                exit(-1);
            }
            break;
        case 'x':
            xc = atof(optarg);
            break;
        case 'y':
            yc = atof(optarg);
            break;
        case 'W':
            W = atof(optarg);
            break;
        case 'H':
            H = atof(optarg);
            break;
        default:
            print_help(args[0]);
            exit(-1);
        }
    }
    if (!jpegfn || !outfn || !text) {
        fprintf(stderr, "No jpeg input or output filename or text.\n");
        print_help(args[0]);
        exit(-1);
    }

    img = cairoutils_read_jpeg(jpegfn, &imW, &imH);
    if (!img) {
        fprintf(stderr, "Failed to read jpeg file %s.\n", jpegfn);
        exit(-1);
    }
    if (xc == -1)
        xc = imW/2;
    if (yc == -1)
        yc = imH/2;
    if (W == 0)
        W = imW;
    if (H == 0)
        H = imH;

    fprintf(stderr, "Image size %i x %i\n", imW, imH);
    fprintf(stderr, "Placing text at center (%g,%g), size (%g,%g)\n", xc, yc, W, H);

    target = cairo_image_surface_create_for_data(img, CAIRO_FORMAT_ARGB32, imW, imH, imW*4);
    cairo = cairo_create(target);
    cairo_set_antialias(cairo, CAIRO_ANTIALIAS_GRAY);
    cairo_set_source_rgb(cairo, r, g, b);
    cairo_select_font_face(cairo, font, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cairo, fontsize);

    // break the text into lines...
    lines = sl_new(4);
    cptr = text;
    while (TRUE) {
        char* nl = strchr(cptr, '\n');
        if (!nl) {
            sl_append(lines, cptr);
            break;
        }
        sl_appendf(lines, "%.*s", (int)((nl - cptr) - 1), cptr);
        cptr = nl + 1;
    }
    // Treat the string "\n" as a newline also.
    for (i=0; i<sl_size(lines); i++) {
        char* line = sl_get(lines, i);
        char* nl = strstr(line, "\\n");
        //fprintf(stderr, "line: %s.  nl: %s\n", line, nl);
        if (!nl)
            continue;
        sl_remove(lines, i);
        sl_insertf(lines, i, "%.*s", (int)(nl - line), line);
        sl_insertf(lines, i+1, "%s", nl + 2);
    }

    /*
     fprintf(stderr, "%i lines:\n", sl_size(lines));
     for (i=0; i<sl_size(lines); i++) {
     char* line = sl_get(lines, i);
     fprintf(stderr, "  >>> %s <<<\n", line);
     }
     */

    for (;;) {
        txtw = txth = 0.0;
        for (i=0; i<sl_size(lines); i++) {
            char* line = sl_get(lines, i);
            cairo_text_extents(cairo, line, &textents);
            txth += textents.height;
            // 
            if (i)
                txth += textents.height * linespacing;
            txtw = MAX(txtw, textents.width);
        }
        fprintf(stderr, "font size %g, txt size = (%g, %g), max size (%g, %g)\n", fontsize, txtw, txth, W, H);
        if (txtw > W || txth > H) {
            double scale = MIN(W / txtw, H / txth);
            fprintf(stderr, "scaling by %g\n", scale);
            fontsize *= scale;
            cairo_set_font_size(cairo, fontsize);
        } else
            break;
    }

    y = yc - txth/2.0;
    for (i=0; i<sl_size(lines); i++) {
        char* line = sl_get(lines, i);
        cairo_text_extents(cairo, line, &textents);
        cairo_move_to(cairo, xc - textents.width / 2.0, y + textents.height/2.0);
        cairo_show_text(cairo, line);
        cairo_stroke(cairo);
        y += textents.height * (1.0 + linespacing);
    }

    if (strcmp(outfn, "-") == 0) {
        res = cairoutils_stream_jpeg(stdout, img, imW, imH);
    } else {
        res = cairoutils_write_jpeg(outfn, img, imW, imH);
    }
    if (res) {
        fprintf(stderr, "Failed to write jpeg to %s.\n", outfn);
        exit(-1);
    }

    return 0;
}
