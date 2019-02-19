/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef PLOTSTUFF_H
#define PLOTSTUFF_H

#include <stdio.h>
#include <cairo.h>

#include "astrometry/keywords.h"
#include "astrometry/sip.h"
#include "astrometry/bl.h"
#include "astrometry/anwcs.h"
#include "astrometry/an-bool.h"

#define PLOTSTUFF_FORMAT_JPG 1
#define PLOTSTUFF_FORMAT_PNG 2
#define PLOTSTUFF_FORMAT_PPM 3
#define PLOTSTUFF_FORMAT_PDF 4
// Save the image as RGBA image "pargs->outimage"
#define PLOTSTUFF_FORMAT_MEMIMG 5
#define PLOTSTUFF_FORMAT_FITS 6

struct plotter;
typedef struct plotter plotter_t;

struct plot_args {
    // the workers
    plotter_t* plotters;
    int NP;

    char* outfn;
    FILE* fout;
    int outformat;

    unsigned char* outimage;

    cairo_t* cairo;
    cairo_surface_t* target;

    cairo_operator_t op;

    // functions to call instead of cairo_move_to / cairo_line_to.
    void (*move_to)(struct plot_args* pargs, double x, double y, void* baton);
    void* move_to_baton;
    void (*line_to)(struct plot_args* pargs, double x, double y, void* baton);
    void* line_to_baton;

    anwcs_t* wcs;

    int W, H;
    float rgba[4];
    float lw; // default: 1
    int marker; // default: circle
    float markersize; // default: 5

    float bg_rgba[4];
    float bg_lw; // default: 3
    int bg_box; // plot a rectangle for text backgrounds.

    float fontsize; // default: 20

    // text alignment
    char halign; // L, R, C
    char valign; // T, B, C

    double label_offset_x;
    double label_offset_y;

    int text_bg_layer;
    int text_fg_layer;
    int marker_fg_layer;

    bl* cairocmds;

    // step size in pixels for drawing curved lines in RA,Dec; default 10
    float linestep;
};
typedef struct plot_args plot_args_t;

typedef void* (*plot_func_init_t)(plot_args_t* args);
typedef int   (*plot_func_init2_t)(plot_args_t* args, void* baton);
typedef int   (*plot_func_command_t)(const char* command, const char* cmdargs, plot_args_t* args, void* baton);
typedef int   (*plot_func_plot_t)(const char* command, cairo_t* cr, plot_args_t* args, void* baton);
typedef void  (*plot_func_free_t)(plot_args_t* args, void* baton);

struct plotter {
    // don't change the order of these fields!
    char* name;
    plot_func_init_t init;
    plot_func_init2_t init2;
    plot_func_command_t command;
    plot_func_plot_t doplot;
    plot_func_free_t free;
    void* baton;
};

//#define DECLARE_PLOTTER(name) plotter_t* plot_ ## name ## _new()
#define DECLARE_PLOTTER(name) void plot_ ## name ## _describe(plotter_t* p)
#define DEFINE_PLOTTER_BODY(pname)              \
    p->name = #pname;                           \
    p->init = plot_ ## pname ## _init;          \
    p->command = plot_ ## pname ## _command;    \
    p->doplot = plot_ ## pname ## _plot;        \
    p->free = plot_ ## pname ## _free;								
		
#define DEFINE_PLOTTER(name) DECLARE_PLOTTER(name) {    \
        DEFINE_PLOTTER_BODY(name)                       \
            }

/*
 #define DEFINE_PLOTTER(name) void plot_ ## name ## _describe(plotter_t* p) { \
 p->name = #name;												\
 p->init = plot_ ## name ## _init;								\
 p->command = plot_ ## name ## _command;							\
 p->doplot = plot_ ## name ## _plot;								\
 p->free = plot_ ## name ## _free;								\
 }
 */

// return PLOTSTUFF_FORMAT_*, or -1 on error
int parse_image_format(const char* fmt);
int guess_image_format_from_filename(const char* fn);
const char* image_format_name_from_code(int code);

int parse_color(const char* color, float* r, float* g, float* b, float* a);
int parse_color_rgba(const char* color, float* rgba);
int cairo_set_color(cairo_t* cairo, const char* color);
void cairo_set_rgba(cairo_t* cairo, const float* rgba);

plot_args_t* plotstuff_new(void);
int plotstuff_init(plot_args_t* plotargs);
int plotstuff_read_and_run_command(plot_args_t* pargs, FILE* f);
int plotstuff_run_command(plot_args_t* pargs, const char* cmd);

void plotstuff_set_text_bg_alpha(plot_args_t* pargs, float alpha);

int plotstuff_plot_layer(plot_args_t* pargs, const char* layer);

void* plotstuff_get_config(plot_args_t* pargs, const char* name);

int plotstuff_set_color(plot_args_t* pargs, const char* name);
int plotstuff_set_bgcolor(plot_args_t* pargs, const char* name);

float plotstuff_get_alpha(const plot_args_t* pargs);

int plotstuff_set_alpha(plot_args_t* pargs, float alpha);

int plotstuff_set_rgba(plot_args_t* pargs, const float* rgba);

int plotstuff_set_rgba2(plot_args_t* pargs, float r, float g, float b, float a);
int plotstuff_set_bgrgba2(plot_args_t* pargs, float r, float g, float b, float a);

int plotstuff_set_marker(plot_args_t* pargs, const char* name);

int plotstuff_set_markersize(plot_args_t* pargs, double ms);

int plotstuff_set_size(plot_args_t* pargs, int W, int H);

// Sets the plot size from the WCS size.
int plotstuff_set_size_wcs(plot_args_t* pargs);

int plotstuff_scale_wcs(plot_args_t* pargs, double scale);

// in deg.
int plotstuff_rotate_wcs(plot_args_t* pargs, double angle);

int plotstuff_set_wcs_box(plot_args_t* pargs, float ra, float dec, float width);

int plotstuff_set_wcs_file(plot_args_t* pargs, const char* fn, int ext);

int plotstuff_set_wcs(plot_args_t* pargs, anwcs_t* wcs);

int plotstuff_set_wcs_tan(plot_args_t* pargs, tan_t* wcs);

int plotstuff_set_wcs_sip(plot_args_t* pargs, sip_t* wcs);

void plotstuff_builtin_apply(cairo_t* cairo, plot_args_t* args);

// Would a marker plotted with the current markersize at x,y appear in the image?
anbool plotstuff_marker_in_bounds(plot_args_t* pargs, double x, double y);

int
ATTRIB_FORMAT(printf,2,3)
    plotstuff_run_commandf(plot_args_t* pargs, const char* fmt, ...);

int plotstuff_output(plot_args_t* pargs);
void plotstuff_free(plot_args_t* pargs);

/* Reset drawing surface with color (0,0,0) and alpha=0 */
void plotstuff_clear(plot_args_t* pargs);


void plotstuff_stack_marker(plot_args_t* pargs, double x, double y);
void plotstuff_stack_arrow(plot_args_t* pargs, double x, double y,
                           double x2, double y2);
void plotstuff_stack_text(plot_args_t* pargs, cairo_t* cairo,
                          const char* txt, double px, double py);
int plotstuff_plot_stack(plot_args_t* pargs, cairo_t* cairo);

void plotstuff_get_maximum_rgba(plot_args_t* pargs,
                                int* p_r, int* p_g, int* p_b, int* p_a);

/// WCS-related stuff:

// in arcsec/pixel
double plotstuff_pixel_scale(plot_args_t* pargs);

// RA,Dec in degrees
// x,y in pixels (cairo coordinates)
// Returns TRUE on success.
anbool plotstuff_radec2xy(plot_args_t* pargs, double ra, double dec,
                          double* p_x, double* p_y);

// RA,Dec in degrees
// x,y in pixels (FITS coordinates)
// Returns TRUE on success.
anbool plotstuff_xy2radec(plot_args_t* pargs, double x, double y,
                          double* pre, double* pdec);

// RA,Dec,radius in deg.
int plotstuff_get_radec_center_and_radius(plot_args_t* pargs, double* pra, double* pdec, double* pradius);

void plotstuff_get_radec_bounds(const plot_args_t* pargs, int stepsize,
                                double* pramin, double* pramax,
                                double* pdecmin, double* pdecmax);

anbool plotstuff_radec_is_inside_image(plot_args_t* pargs, double ra, double dec);

int plotstuff_line_constant_ra(plot_args_t* pargs, double ra, double dec1, double dec2,
                               anbool startwithmove);
int plotstuff_line_constant_dec(plot_args_t* pargs, double dec, double ra1, double ra2);
int plotstuff_line_constant_dec2(plot_args_t* pargs, double dec,
                                 double ra1, double ra2, double stepra);

int plotstuff_text_xy(plot_args_t* pargs, double ra, double dec, const char* label);
int plotstuff_text_radec(plot_args_t* pargs, double ra, double dec, const char* label);
int plotstuff_move_to_radec(plot_args_t* pargs, double ra, double dec);
int plotstuff_line_to_radec(plot_args_t* pargs, double ra, double dec);
int plotstuff_close_path(plot_args_t* pargs);
int plotstuff_stroke(plot_args_t* pargs);
int plotstuff_fill(plot_args_t* pargs);
int plotstuff_stroke_preserve(plot_args_t* pargs);
int plotstuff_fill_preserve(plot_args_t* pargs);

void plotstuff_move_to(plot_args_t* pargs, double x, double y);
void plotstuff_line_to(plot_args_t* pargs, double x, double y);

void plotstuff_marker(plot_args_t* pargs, double x, double y);
int plotstuff_marker_radec(plot_args_t* pargs, double ra, double dec);

int plotstuff_append_doubles(const char* str, dl* lst);

void plotstuff_set_dashed(plot_args_t* pargs, double dashlen);
void plotstuff_set_solid(plot_args_t* pargs);

#endif
