/*
 This file is part of the Astrometry.net suite.
 Copyright 2007 Michael Blanton, Keir Mierle, David W. Hogg,
 Sam Roweis and Dustin Lang.
 
 The Astrometry.net suite is free software; you can redistribute it
 and/or modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, version 2.

 The Astrometry.net suite is distributed in the hope that it will be
 useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with the Astrometry.net suite ; if not, write to the Free
 Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 02110-1301 USA
 */
#ifndef SIMPLEXY2_H
#define SIMPLEXY2_H

struct simplexy_t {
    /******
     Inputs
     ******/
    float *image;
    unsigned char* image_u8;
    int nx;
    int ny;
    /* gaussian psf width; 1 is usually fine */
    float dpsf;
    /* significance to keep; 8 is usually fine */
    float plim;
    /* closest two peaks can be; 1 is usually fine */
    float dlim;
    /* saddle difference (in sig); 3 is usually fine */
    float saddle;
    /* maximum number of peaks per object; 1000 */
    int maxper;
    /* maximum number of peaks total; 100000 */
    int maxnpeaks;
    /* maximum size for extended objects: 150 */
    int maxsize;
    /* size for sliding sky estimation box */
    int halfbox;

    /******
     Outputs
     ******/
    float sigma;
    float *x;
    float *y;
    float *flux;
    float *background;
    int npeaks;
    int verbose;

    /******
     Internal
     ******/
    float* simage;
    unsigned char* simage_u8;

    int* oimage;
    float* smooth;
};
typedef struct simplexy_t simplexy_t;

int simplexy2(simplexy_t* s);

#endif
