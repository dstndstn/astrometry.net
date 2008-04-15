#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <sys/mman.h>

#include "tilerender.h"
#include "render_solid.h"

static void logmsg(char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "render_solid: ");
    vfprintf(stderr, format, args);
    va_end(args);
}

int render_solid(unsigned char* img, render_args_t* args) {
    int i, j;

    logmsg("render_solid: filling with RGBA=(0,0,0,255)\n");

    for (j=0; j<args->H; j++) {
        for (i=0; i<args->W; i++) {
            uchar* pix = pixel(i, j, img, args);
            pix[0] = 0;
            pix[1] = 0;
            pix[2] = 0;
            pix[3] = 255;
        }
    }
    return 0;
}

