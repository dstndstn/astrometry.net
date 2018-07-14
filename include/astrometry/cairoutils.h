/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef CAIRO_UTILS_H
#define CAIRO_UTILS_H

#include <stdio.h>

#include <cairo.h>

/**
 A cairo_write_func_t for use with, eg, cairo_pdf_surface_create_for_stream.
 The "closure" arg must be a FILE*.
 */
cairo_status_t cairoutils_file_write_func(void *closure,
                                          const unsigned char *data,
                                          unsigned int length);

/**
 Reports any cairo error for the given surface.  Returns 0 for ok, -1 on error.
 */
int cairoutils_surface_status_errors(cairo_surface_t* surf);

int cairoutils_cairo_status_errors(cairo_t* c);

void cairoutils_draw_path(cairo_t* c, const double* xy, int N);

void cairoutils_argb32_to_rgba(unsigned char* img, int W, int H);

void cairoutils_argb32_to_rgba_2(const unsigned char* inimg,
                                 unsigned char* outimg, int W, int H);

void cairoutils_argb32_to_rgba_flip(const unsigned char* inimg,
                                    unsigned char* outimg, int W, int H);

void cairoutils_rgba_to_argb32(unsigned char* img, int W, int H);

void cairoutils_rgba_to_argb32_2(const unsigned char* inimg,
                                 unsigned char* outimg, int W, int H);

void cairoutils_rgba_to_argb32_flip(const unsigned char* inimg,
                                    unsigned char* outimg, int W, int H);

void cairoutils_premultiply_alpha_rgba(unsigned char* img, int W, int H);

/**
 All the following cairoutils_read_* function return a newly-allocated image buffer
 of size W * H * 4, containing R,G,B, and alpha.
 */
unsigned char* cairoutils_read_png_stream(FILE* fid, int* pW, int *pH);

unsigned char* cairoutils_read_jpeg_stream(FILE* fid, int* pW, int* pH);

unsigned char* cairoutils_read_png(const char* fn, int* pW, int *pH);

unsigned char* cairoutils_read_jpeg(const char* fn, int* pW, int* pH);

void cairoutils_fake_ppm_init(void);

// You must call ppm_init() or (preferably) cairoutils_fake_ppm_init()
unsigned char* cairoutils_read_ppm(const char* infn, int* pW, int* pH);

unsigned char* cairoutils_read_ppm_stream(FILE* fid, int* pW, int* pH);

int cairoutils_write_ppm(const char* outfn, unsigned char* img, int W, int H);

int cairoutils_write_png(const char* outfn, unsigned char* img, int W, int H);

int cairoutils_write_jpeg(const char* outfn, unsigned char* img, int W, int H);

int cairoutils_stream_ppm(FILE* fout, unsigned char* img, int W, int H);

int cairoutils_stream_png(FILE* fout, unsigned char* img, int W, int H);

int cairoutils_stream_jpeg(FILE* fout, unsigned char* img, int W, int H);

int cairoutils_parse_color(const char* color, float* r, float* g, float* b);

// Parses a space-separated list of floating-point rgb(a) values.
// Parses alpha if "a" is non-null and str contains four terms.
int cairoutils_parse_rgba(const char* str, float* r, float* g, float* b, float* a);

const char* cairoutils_get_color_name(int i);

int cairoutils_parse_marker(const char* name);

#define CAIROUTIL_MARKER_CIRCLE 0
#define CAIROUTIL_MARKER_CROSSHAIR 1
#define CAIROUTIL_MARKER_SQUARE 2
#define CAIROUTIL_MARKER_DIAMOND 3
#define CAIROUTIL_MARKER_X 4
#define CAIROUTIL_MARKER_XCROSSHAIR 5

void cairoutils_draw_marker(cairo_t* cairo, int id, double x, double y, double radius);

const char* cairoutils_get_marker_name(int i);

void cairoutils_print_color_names(const char* prefix);
void cairoutils_print_marker_names(const char* prefix);

#endif

