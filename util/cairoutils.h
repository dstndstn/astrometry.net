/*
  This file is part of the Astrometry.net suite.
  Copyright 2007 Dustin Lang, Keir Mierle and Sam Roweis.

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

#ifndef CAIRO_UTILS_H
#define CAIRO_UTILS_H

#include <stdio.h>

#include <cairo.h>

void cairoutils_argb32_to_rgba(unsigned char* img, int W, int H);

void cairoutils_rgba_to_argb32(unsigned char* img, int W, int H);

unsigned char* cairoutils_read_png_stream(FILE* fid, int* pW, int *pH);

unsigned char* cairoutils_read_jpeg_stream(FILE* fid, int* pW, int* pH);

unsigned char* cairoutils_read_png(const char* fn, int* pW, int *pH);

unsigned char* cairoutils_read_jpeg(const char* fn, int* pW, int* pH);

void cairoutils_fake_ppm_init();

// You must call ppm_init()
unsigned char* cairoutils_read_ppm(const char* infn, int* pW, int* pH);

unsigned char* cairoutils_read_ppm_stream(FILE* fid, int* pW, int* pH);

int cairoutils_write_ppm(const char* outfn, unsigned char* img, int W, int H);

int cairoutils_write_png(const char* outfn, unsigned char* img, int W, int H);

int cairoutils_write_jpeg(const char* outfn, unsigned char* img, int W, int H);

int cairoutils_stream_ppm(FILE* fout, unsigned char* img, int W, int H);

int cairoutils_stream_png(FILE* fout, unsigned char* img, int W, int H);

int cairoutils_stream_jpeg(FILE* fout, unsigned char* img, int W, int H);

int cairoutils_parse_color(const char* color, float* r, float* g, float* b);

const char* cairoutils_get_color_name(int i);

int cairoutils_parse_marker(const char* name);

void cairoutils_draw_marker(cairo_t* cairo, int id, double x, double y, double radius);

const char* cairoutils_get_marker_name(int i);

void cairoutils_print_color_names(const char* prefix);
void cairoutils_print_marker_names(const char* prefix);

#endif

