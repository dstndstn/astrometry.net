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

#ifndef TILERENDER_H
#define TILERENDER_H

#include <cairo.h>

#include "starutil.h"
#include "mathutil.h"
#include "merc.h"
#include "bl.h"

struct render_args {
	// In degrees
	double ramin;
	double ramax;
	double decmin;
	double decmax;

	// In Mercator units: [0,1].
	double xmercmin;
	double xmercmax;
	double ymercmin;
	double ymercmax;

	// Mercator units per pixel in X,Y directions.
	double xmercperpixel;
	double ymercperpixel;

	double xpixelpermerc;
	double ypixelpermerc;

	int zoomlevel;

    // Don't render PNG; spit out raw floats.
    int makerawfloatimg;
    float* rawfloatimg;

	// Image size in pixels.
	int W;
	int H;

    // The current layer we're rendering.
    char* currentlayer;

    // render_images / render_boundary:
    char* filelist;
    bool density;

    // render_boundary
    char* colorlist;

	// Args for render_image:
	/*
	  char* imagefn;
	  char* imwcsfn;
	*/
	sl* imagefns;
	sl* imwcsfns;

	// Args for render_tycho:
	char* tycho_mkdt;
	double colorcor;
	bool arc;
	bool sqrt;
	bool arith;
	double gain;
	double nlscale;

	// Args for render_usnob:
	char* cmap;
	bool nopre; // don't use pre-rendered tiles (useful when you're trying to *make* pre-rendered tiles)

    // Assume the prerendered tiles are backwards (ie were rendered before the RA/merc flip)
    bool pre_backward;

    char* version;

	// Args for render_rdls
	sl* rdlsfns;
	sl* rdlscolors;
	il* Nstars;
	il* fieldnums;

	// Args for render_boundary
	char* wcsfn;
	double linewidth;
	double dashbox;
	bool zoomright;
	bool zoomdown;
	char* ubstyle;

	// Args for render_healpixes
	int nside;

	char* constfn;

    // caching
    char* cachedir;

    // generic argument-passing:
    // argument filenames
    sl* argfilenames;
    // a list of all the lines in all the files.
    sl* arglist;

};
typedef struct render_args render_args_t;

typedef int (*render_func_t)(unsigned char* dest_img, render_args_t* args);
typedef int (*render_cairo_func_t)(cairo_t* cr, render_args_t* args);

void get_string_args_of_type(render_args_t* args, const char* prefix, sl* lst);

void get_string_args_of_types(render_args_t* args, const char* prefixes[], int Nprefixes, sl* lst, sl* matched_prefixes);

void get_double_args_of_type(render_args_t* args, const char* prefix, dl* lst);

// Returns the first argument of the given name.
double get_double_arg_of_type(render_args_t* args, const char* name, double def);

int get_int_arg(const char* arg, int def);

double get_double_arg(const char* arg, double def);

void get_double_args(const char* arg, dl* lst);

int parse_color(char c, double* p_red, double* p_green, double* p_blue);

// to RA in degrees
double pixel2ra(double pix, render_args_t* args);

// to DEC in degrees
double pixel2dec(double pix, render_args_t* args);

int xmerc2pixel(double x, render_args_t* args);

int ymerc2pixel(double y, render_args_t* args);

double xmerc2pixelf(double x, render_args_t* args);

double ymerc2pixelf(double y, render_args_t* args);

// RA in degrees
int ra2pixel(double ra, render_args_t* args);

// DEC in degrees
int dec2pixel(double dec, render_args_t* args);

// RA in degrees
double ra2pixelf(double ra, render_args_t* args);

// DEC in degrees
double dec2pixelf(double dec, render_args_t* args);

double xpixel2mercf(double pix, render_args_t* args);
double ypixel2mercf(double pix, render_args_t* args);

int in_image(int x, int y, render_args_t* args);

// Like in_image, but with a margin around the outside.
int in_image_margin(int x, int y, int margin, render_args_t* args);

// void put_pixel(int x, int y, uchar r, uchar g, uchar b, uchar a, render_args_t* args, uchar* img);

uchar* pixel(int x, int y, uchar* img, render_args_t* args);

void draw_segmented_line(double ra1, double dec1,
						 double ra2, double dec2,
						 int SEGS,
						 cairo_t* cairo, render_args_t* args);

// draw a line in Mercator space, handling wrap-around if necessary.
void draw_line_merc(double mx1, double my1, double mx2, double my2,
					cairo_t* cairo, render_args_t* args);

void* cache_load(render_args_t* args,
                 const char* cachedomain, const char* key, int* length);

int cache_save(render_args_t* args,
               const char* cachedomain, const char* key,
               const void* data, int length);

#endif
