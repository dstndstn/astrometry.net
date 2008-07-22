/*
   This file is part of the Astrometry.net suite.
   Copyright 2007 Keir Mierle and Dustin Lang.

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
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

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

