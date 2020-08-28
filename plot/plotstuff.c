/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

// Avoid *nasty* problem when 'bool' gets redefined (by ppm.h) to be 4 bytes!
#include "an-bool.h"

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <cairo.h>
#include <cairo-pdf.h>

#include "os-features.h"
#include "plotstuff.h"
#include "plotfill.h"
#include "plotxy.h"
#include "plotimage.h"
#include "plotannotations.h"
#include "plotgrid.h"
#include "plotoutline.h"
#include "plotindex.h"
#include "plotradec.h"
#include "plothealpix.h"
#include "plotmatch.h"

#include "sip_qfits.h"
#include "sip-utils.h"
#include "sip.h"
#include "cairoutils.h"
#include "starutil.h"
#include "ioutils.h"
#include "log.h"
#include "errors.h"
#include "anwcs.h"


enum cmdtype {
    CIRCLE,
    TEXT,
    LINE,
    RECTANGLE,
    ARROW,
    MARKER,
    POLYGON,
};
typedef enum cmdtype cmdtype;

struct cairocmd {
    cmdtype type;
    int layer;
    double x, y;
    float rgba[4];
    // CIRCLE
    double radius;
    // TEXT
    char* text;
    // LINE / RECTANGLE / ARROW
    double x2, y2;
    // MARKER
    int marker;
    double markersize;
    // POLYGON
    dl* xy;
    anbool fill;
};
typedef struct cairocmd cairocmd_t;

static void get_text_position(plot_args_t* pargs, cairo_t* cairo,
                              const char* txt, double* px, double* py);

plot_args_t* plotstuff_new() {
    plot_args_t* pargs = calloc(1, sizeof(plot_args_t));
    plotstuff_init(pargs);
    return pargs;
}

void plotstuff_clear(plot_args_t* pargs) {
    cairo_operator_t op;
    assert(pargs->cairo);
    op = cairo_get_operator(pargs->cairo);
    cairo_set_operator(pargs->cairo, CAIRO_OPERATOR_CLEAR);
    cairo_paint(pargs->cairo);
    cairo_set_operator(pargs->cairo, op);
}

void plotstuff_move_to(plot_args_t* pargs, double x, double y) {
    if (pargs->move_to)
        pargs->move_to(pargs, x, y, pargs->move_to_baton);
    else {
        assert(pargs->cairo);
        cairo_move_to(pargs->cairo, x, y);
    }
}

void plotstuff_line_to(plot_args_t* pargs, double x, double y) {
    if (pargs->line_to)
        pargs->line_to(pargs, x, y, pargs->line_to_baton);
    else {
        assert(pargs->cairo);
        cairo_line_to(pargs->cairo, x, y);
    }
}

int plotstuff_rotate_wcs(plot_args_t* pargs, double angle) {
    if (!pargs->wcs) {
        ERROR("No WCS has been set");
        return -1;
    }
    return anwcs_rotate_wcs(pargs->wcs, angle);
}

int plotstuff_get_radec_center_and_radius(plot_args_t* pargs,
                                          double* p_ra, double* p_dec, double* p_radius) {
    int rtn;
    if (!pargs->wcs)
        return -1;
    rtn = anwcs_get_radec_center_and_radius(pargs->wcs, p_ra, p_dec, p_radius);
    if (rtn)
        return rtn;
    if (p_radius && *p_radius == 0.0) {
        // HACK -- get approximate scale, using plot size.
        *p_radius = arcsec2deg(anwcs_pixel_scale(pargs->wcs) * hypot(pargs->W, pargs->H)/2.0);
    }
    return rtn;
}

int plotstuff_append_doubles(const char* str, dl* lst) {
    int i;
    sl* strs = sl_split(NULL, str, " ");
    for (i=0; i<sl_size(strs); i++)
        dl_append(lst, atof(sl_get(strs, i)));
    sl_free2(strs);
    return 0;
}

int plotstuff_line_constant_ra(plot_args_t* pargs, double ra, double dec1, double dec2,
                               anbool startwithmove) {
    double decstep;
    double dec;
    double s;
    double pixscale;
    anbool lastok = FALSE;
    if (!startwithmove)
        lastok = TRUE;
    assert(pargs->wcs);
    pixscale = anwcs_pixel_scale(pargs->wcs);
    assert(pixscale > 0.0);
    decstep = arcsec2deg(pixscale * pargs->linestep);
    logverb("plotstuff_line_constant_ra: RA=%g, Dec=[%g,%g], pixscale %g, decstep %g\n",
            ra, dec1, dec2, anwcs_pixel_scale(pargs->wcs), decstep);
    //printf("plotstuff_line_constant_ra: RA=%g, Dec=[%g,%g], pixscale %g, decstep %g\n",
    //ra, dec1, dec2, anwcs_pixel_scale(pargs->wcs), decstep);
    s = 1.0;
    if (dec1 > dec2)
        s = -1;
    for (dec=dec1; (s*dec)<=(s*dec2); dec+=(decstep*s)) {
        double x, y;
        //logverb("  line_constant_ra: RA,Dec %g,%g\n", ra, dec);
        //printf("  line_constant_ra: RA,Dec %g,%g\n", ra, dec);
        if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y)) {
            printf("  bad xy\n");
            lastok = FALSE;
            continue;
        }
        //printf("  x,y %.1f, %.1f\n", x, y);
        if (lastok)
            plotstuff_line_to(pargs, x, y);
        else
            plotstuff_move_to(pargs, x, y);
        lastok = TRUE;
    }
    return 0;
}

int plotstuff_line_constant_dec(plot_args_t* pargs, double dec, double ra1, double ra2) {
    double rastep;
    double ra;
    double f;
    double s;
    assert(pargs->wcs);
    rastep = arcsec2deg(anwcs_pixel_scale(pargs->wcs) * pargs->linestep);
    f = cos(deg2rad(dec));
    rastep /= MAX(0.1, f);
    s = 1.0;
    if (ra1 > ra2)
        s = -1.0;
    for (ra=ra1; (s*ra)<=(s*ra2); ra+=(rastep*s)) {
        double x, y;
        if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y))
            continue;
        if (ra == ra1)
            plotstuff_move_to(pargs, x, y);
        else
            plotstuff_line_to(pargs, x, y);
    }
    return 0;
}

static double normra(double ra) {
    while (ra < 0.)
        ra += 360.;
    while (ra > 360.)
        ra -= 360.;
    return ra;
}

int plotstuff_line_constant_dec2(plot_args_t* pargs, double dec, double ra1, double ra2, double rastep) {
    double ra;
    int n;
    int done = 0;
    ra1 = normra(ra1);
    ra2 = normra(ra2);
    assert(pargs->wcs);
    for (ra=ra1, n=0; n<1000000; n++) {
        double x, y;
        double ranext;
        ra = normra(ra);
        if (anwcs_radec2pixelxy(pargs->wcs, ra, dec, &x, &y))
            continue;
        if (n == 0)
            plotstuff_move_to(pargs, x, y);
        else
            plotstuff_line_to(pargs, x, y);
        if (done)
            break;
        ranext = ra + rastep;
        // will the next step take us past ra2?
        if (MIN(ra, ranext) < ra2 && ra2 < MAX(ra, ranext)) {
            ra = ra2;
            done = 1;
        } else
            ra = ranext;
    }
    return 0;
}

int plotstuff_text_radec(plot_args_t* pargs, double ra, double dec, const char* label) {
    double x,y;
    if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y)) {
        ERROR("Failed to convert RA,Dec (%g,%g) to pixel position in plot_text_radec\n", ra, dec);
        return -1;
    }
    assert(pargs->cairo);
    //plotstuff_stack_text(pargs, pargs->cairo, label, x, y);
    get_text_position(pargs, pargs->cairo, label, &x, &y);
    plotstuff_move_to(pargs, x, y);
    cairo_show_text(pargs->cairo, label);
    return 0;
}

int plotstuff_text_xy(plot_args_t* pargs, double x, double y, const char* label) {
    assert(pargs->cairo);
    get_text_position(pargs, pargs->cairo, label, &x, &y);
    plotstuff_move_to(pargs, x, y);
    cairo_show_text(pargs->cairo, label);
    return 0;
}

static int moveto_lineto_radec(plot_args_t* pargs, double ra, double dec, anbool move) {
    double x,y;
    if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y)) {
        ERROR("Failed to convert RA,Dec (%g,%g) to pixel position in plot_text_radec\n", ra, dec);
        return -1;
    }
    assert(pargs->cairo);
    (move ? plotstuff_move_to : plotstuff_line_to)(pargs, x, y);
    return 0;
}

int plotstuff_move_to_radec(plot_args_t* pargs, double ra, double dec) {
    assert(pargs->cairo);
    plotstuff_builtin_apply(pargs->cairo, pargs);
    return moveto_lineto_radec(pargs, ra, dec, TRUE);
}

int plotstuff_line_to_radec(plot_args_t* pargs, double ra, double dec) {
    return moveto_lineto_radec(pargs, ra, dec, FALSE);
}

int plotstuff_close_path(plot_args_t* pargs) {
    assert(pargs->cairo);
    cairo_close_path(pargs->cairo);
    return 0;
}

int plotstuff_fill(plot_args_t* pargs) {
    assert(pargs->cairo);
    cairo_fill(pargs->cairo);
    return 0;
}
int plotstuff_stroke(plot_args_t* pargs) {
    assert(pargs->cairo);
    cairo_stroke(pargs->cairo);
    return 0;
}

int plotstuff_fill_preserve(plot_args_t* pargs) {
    assert(pargs->cairo);
    cairo_fill_preserve(pargs->cairo);
    return 0;
}
int plotstuff_stroke_preserve(plot_args_t* pargs) {
    assert(pargs->cairo);
    cairo_stroke_preserve(pargs->cairo);
    return 0;
}

void plotstuff_set_dashed(plot_args_t* pargs, double dashlen) {
    assert(pargs->cairo);
    cairo_set_dash(pargs->cairo, &dashlen, 1, 0);
}

void plotstuff_set_solid(plot_args_t* pargs) {
    assert(pargs->cairo);
    cairo_set_dash(pargs->cairo, NULL, 0, 0);
}

int parse_color(const char* color, float* r, float* g, float* b, float* a) {
    if (a) *a = 1.0;
    return (cairoutils_parse_rgba(color, r, g, b, a) &&
            cairoutils_parse_color(color, r, g, b));
}

int parse_color_rgba(const char* color, float* rgba) {
    return parse_color(color, rgba, rgba+1, rgba+2, rgba+3);
}

void cairo_set_rgba(cairo_t* cairo, const float* rgba) {
    cairo_set_source_rgba(cairo, rgba[0], rgba[1], rgba[2], rgba[3]);
}

int cairo_set_color(cairo_t* cairo, const char* color) {
    float rgba[4];
    int res;
    res = parse_color_rgba(color, rgba);
    if (res) {
        ERROR("Failed to parse color \"%s\"", color);
        return res;
    }
    cairo_set_rgba(cairo, rgba);
    return res;
}

void plotstuff_builtin_apply(cairo_t* cairo, plot_args_t* args) {
    //printf("Set rgba %.2f, %.2f, %.2f, %.2f\n", args->rgba[0], args->rgba[1],
    //args->rgba[2], args->rgba[3]);
    cairo_set_rgba(cairo, args->rgba);
    cairo_set_line_width(cairo, args->lw);
    cairo_set_operator(cairo, args->op);
    cairo_set_font_size(cairo, args->fontsize);
}

void plotstuff_set_text_bg_alpha(plot_args_t* pargs, float alpha) {
    pargs->bg_rgba[3] = alpha;
}

static void* plot_builtin_init(plot_args_t* args) {
    parse_color_rgba("gray", args->rgba);
    parse_color_rgba("black", args->bg_rgba);
    args->text_bg_layer = 2;
    args->text_fg_layer = 3;
    args->marker_fg_layer = 3;
    args->bg_lw = 3.0;
    args->lw = 1.0;
    args->marker = CAIROUTIL_MARKER_CIRCLE;
    args->markersize = 5.0;
    args->linestep = 10;
    args->op = CAIRO_OPERATOR_OVER;
    args->fontsize = 20;
    args->halign = 'C';
    args->valign = 'B';
    args->cairocmds = bl_new(256, sizeof(cairocmd_t));
    args->label_offset_x = 10.0;
    args->label_offset_y =  5.0;
    return NULL;
}

static int plot_builtin_init2(plot_args_t* pargs, void* baton) {
    plotstuff_builtin_apply(pargs->cairo, pargs);
    // Inits that aren't in "plot_builtin"
    cairo_set_antialias(pargs->cairo, CAIRO_ANTIALIAS_GRAY);
    return 0;
}

int plotstuff_set_markersize(plot_args_t* pargs, double ms) {
    pargs->markersize = ms;
    return 0;
}

int plotstuff_set_marker(plot_args_t* pargs, const char* name) {
    int m = cairoutils_parse_marker(name);
    if (m == -1) {
        ERROR("Failed to parse plot_marker \"%s\"", name);
        return -1;
    }
    pargs->marker = m;
    return 0;
}

int plotstuff_set_size(plot_args_t* pargs, int W, int H) {
    pargs->W = W;
    pargs->H = H;
    return 0;
}

int plotstuff_scale_wcs(plot_args_t* pargs, double scale) {
    if (!pargs->wcs) {
        ERROR("No WCS has been set");
        return -1;
    }
    return anwcs_scale_wcs(pargs->wcs, scale);
}

int plotstuff_set_wcs_file(plot_args_t* pargs, const char* filename, int ext) {
    anwcs_t* wcs = anwcs_open(filename, ext);
    if (!wcs) {
        ERROR("Failed to read WCS file \"%s\", extension %i", filename, ext);
        return -1;
    }
    return plotstuff_set_wcs(pargs, wcs);
}

int plotstuff_set_wcs_sip(plot_args_t* pargs, sip_t* wcs) {
    anwcs_t* anwcs = anwcs_new_sip(wcs);
    return plotstuff_set_wcs(pargs, anwcs);
}

int plotstuff_set_wcs_tan(plot_args_t* pargs, tan_t* wcs) {
    anwcs_t* anwcs = anwcs_new_tan(wcs);
    return plotstuff_set_wcs(pargs, anwcs);
}

int plotstuff_set_wcs(plot_args_t* pargs, anwcs_t* wcs) {
    if (pargs->wcs) {
        anwcs_free(pargs->wcs);
    }
    pargs->wcs = wcs;
    return 0;
}

int plotstuff_set_wcs_box(plot_args_t* pargs, float ra, float dec, float width) {
    logverb("Setting WCS to a box centered at (%g,%g) with width %g deg.\n", ra, dec, width);
    anwcs_t* wcs = anwcs_create_box_upsidedown(ra, dec, width, pargs->W, pargs->H);
    return plotstuff_set_wcs(pargs, wcs);
}

static int plot_builtin_command(const char* cmd, const char* cmdargs,
                                plot_args_t* pargs, void* baton) {
    if (streq(cmd, "plot_color")) {
        if (parse_color_rgba(cmdargs, pargs->rgba)) {
            ERROR("Failed to parse plot_color: \"%s\"", cmdargs);
            return -1;
        }
    } else if (streq(cmd, "plot_bgcolor")) {
        if (parse_color_rgba(cmdargs, pargs->bg_rgba)) {
            ERROR("Failed to parse plot_bgcolor: \"%s\"", cmdargs);
            return -1;
        }
    } else if (streq(cmd, "plot_fontsize")) {
        pargs->fontsize = atof(cmdargs);
    } else if (streq(cmd, "plot_alpha")) {
        if (plotstuff_set_alpha(pargs, atof(cmdargs))) {
            ERROR("Failed to set alpha");
            return -1;
        }
    } else if (streq(cmd, "plot_op")) {
        if (streq(cmdargs, "add")) {
            pargs->op = CAIRO_OPERATOR_ADD;
        } else if (streq(cmdargs, "reset")) {
            pargs->op = CAIRO_OPERATOR_OVER;
        } else {
            ERROR("Didn't understand op: %s", cmdargs);
            return -1;
        }
    } else if (streq(cmd, "plot_lw")) {
        pargs->lw = atof(cmdargs);
    } else if (streq(cmd, "plot_bglw")) {
        pargs->bg_lw = atof(cmdargs);
    } else if (streq(cmd, "plot_marker")) {
        if (plotstuff_set_marker(pargs, cmdargs)) {
            return -1;
        }
    } else if (streq(cmd, "plot_markersize")) {
        pargs->markersize = atof(cmdargs);
    } else if (streq(cmd, "plot_size")) {
        int W, H;
        if (sscanf(cmdargs, "%i %i", &W, &H) != 2) {
            ERROR("Failed to parse plot_size args \"%s\"", cmdargs);
            return -1;
        }
        plotstuff_set_size(pargs, W, H);
    } else if (streq(cmd, "plot_wcs")) {
        if (plotstuff_set_wcs_file(pargs, cmdargs, 0)) {
            return -1;
        }
    } else if (streq(cmd, "plot_wcs_box")) {
        float ra, dec, width;
        if (sscanf(cmdargs, "%f %f %f", &ra, &dec, &width) != 3) {
            ERROR("Failed to parse plot_wcs_box args \"%s\"", cmdargs);
            return -1;
        }
        if (plotstuff_set_wcs_box(pargs, ra, dec, width)) {
            return -1;
        }
    } else if (streq(cmd, "plot_wcs_setsize")) {
        assert(pargs->wcs);
        plotstuff_set_size_wcs(pargs);
    } else if (streq(cmd, "plot_label_radec")) {
        assert(pargs->wcs);
        double ra, dec;
        int nc;
        const char* label;
        if (sscanf(cmdargs, "%lf %lf %n", &ra, &dec, &nc) != 3) {
            ERROR("Failed to parse plot_label_radec args \"%s\"", cmdargs);
            return -1;
        }
        label = cmdargs + nc;
        return plotstuff_text_radec(pargs, ra, dec, label);
    } else {
        ERROR("Did not understand command: \"%s\"", cmd);
        return -1;
    }
    if (pargs->cairo)
        plotstuff_builtin_apply(pargs->cairo, pargs);
    return 0;
}

int plotstuff_set_size_wcs(plot_args_t* pargs) {
    assert(pargs->wcs);
    return plotstuff_set_size(pargs, (int)ceil(anwcs_imagew(pargs->wcs)), (int)ceil(anwcs_imageh(pargs->wcs)));
}

int plot_builtin_plot(const char* command, cairo_t* cairo, plot_args_t* pargs, void* baton) {
    //plotstuff_plot_stack(pargs, cairo);
    return 0;
}

static void cairocmd_init(cairocmd_t* cmd) {
    if (!cmd)
        return;
    memset(cmd, 0, sizeof(cairocmd_t));
    //cmd->xy = dl_new(32);
}

static void cairocmd_clear(cairocmd_t* cmd) {
    if (!cmd)
        return;
    free(cmd->text);
    cmd->text = NULL;
    if (cmd->xy)
        dl_free(cmd->xy);
    cmd->xy = NULL;
}

static void add_cmd(plot_args_t* pargs, cairocmd_t* cmd) {
    bl_append(pargs->cairocmds, cmd);
}

static void set_cmd_args(plot_args_t* pargs, cairocmd_t* cmd) {
    cmd->marker = pargs->marker;
    cmd->markersize = pargs->markersize;
    memcpy(cmd->rgba, pargs->rgba, sizeof(cmd->rgba));
}

anbool plotstuff_marker_in_bounds(plot_args_t* pargs, double x, double y) {
    double margin = pargs->markersize;
    return (x >= -margin && x <= (pargs->W + margin) &&
            y >= -margin && y <= (pargs->H + margin));
}

void plotstuff_stack_marker(plot_args_t* pargs, double x, double y) {
    cairocmd_t cmd;
    cairocmd_init(&cmd);
    set_cmd_args(pargs, &cmd);
    // BG marker?
    cmd.layer = pargs->marker_fg_layer;
    cmd.type = MARKER;
    // FIXME -- handle cairo half-pixel issues here?
    cmd.x = x + 0.5;
    cmd.y = y + 0.5;
    add_cmd(pargs, &cmd);
}

void plotstuff_stack_arrow(plot_args_t* pargs, double x, double y,
                           double x2, double y2) {
    cairocmd_t cmd;
    cairocmd_init(&cmd);
    // BG?
    set_cmd_args(pargs, &cmd);
    cmd.layer = pargs->marker_fg_layer;
    cmd.type = ARROW;
    cmd.x = x;
    cmd.y = y;
    cmd.x2 = x2;
    cmd.y2 = y2;
    add_cmd(pargs, &cmd);
}

static void get_text_position(plot_args_t* pargs, cairo_t* cairo,
                              const char* txt, double* px, double* py) {
    cairo_text_extents_t textents;
    double l = 0.0,r,t = 0.0,b;
    double margin = 2.0;
    double x, y;
    x = *px;
    y = *py;

    x += pargs->label_offset_x;
    y += pargs->label_offset_y;

    cairo_text_extents(cairo, txt, &textents);

    switch (pargs->halign) {
    case 'L':
        l = x + textents.x_bearing;
        break;
    case 'C':
        l = x + textents.x_bearing - 0.5*textents.width;
        break;
    case 'R':
        l = x + textents.x_bearing - textents.width;
        break;
    }
    x = l;
    r = l + textents.width + textents.x_bearing;

    switch (pargs->valign) {
    case 'T':
        t = y + textents.y_bearing + textents.height;
        //y -= (0.5 * textents.y_bearing);
        break;
    case 'C':
        t = y + textents.y_bearing + 0.5*textents.height;
        break;
    case 'B':
        t = y + textents.y_bearing;
        break;
    }
    b = t + textents.height;
    y = b;

    l -= margin;
    t -= margin;
    r += margin + 1;
    b += margin + 1;

    // move text away from the edges of the image.
    if (l < 0) {
        x += -l;
        l = 0;
    }
    if (t < 0) {
        y += -t;
        t = 0;
    }
    if (r > pargs->W) {
        x -= (r - pargs->W);
        r = pargs->W;
    }
    if (b > pargs->H) {
        y -= (b - pargs->H);
        b = pargs->H;
    }

    *px = x;
    *py = y;
}

void plotstuff_stack_text(plot_args_t* pargs, cairo_t* cairo,
                          const char* txt, double px, double py) {
    cairocmd_t cmd;
    cairocmd_init(&cmd);
    set_cmd_args(pargs, &cmd);
    get_text_position(pargs, cairo, txt, &px, &py);
    cmd.type = TEXT;

    if (pargs->bg_rgba[3] > 0) {
        int dx, dy;
        logverb("Background text RGB [%g, %g, %g] alpha %g\n",
                pargs->bg_rgba[0], pargs->bg_rgba[1], 
                pargs->bg_rgba[2], pargs->bg_rgba[3]);
        cmd.layer = pargs->text_bg_layer;
        memcpy(cmd.rgba, pargs->bg_rgba, sizeof(cmd.rgba));

        if (pargs->bg_box) {
            // Plot a rectangle behind the text
            cairo_text_extents_t textents;
            cairo_text_extents(cairo, txt, &textents);
            cmd.type = RECTANGLE;
            cmd.x = px + textents.x_bearing;
            cmd.y = py + textents.y_bearing;
            cmd.x2 = cmd.x + textents.width;
            cmd.y2 = py + textents.y_bearing + textents.height;
            cmd.fill = TRUE;
            add_cmd(pargs, &cmd);
            cmd.type = TEXT;
        } else {
            // Plot bg-color text behind
            for (dy=-1; dy<=1; dy++) {
                for (dx=-1; dx<=1; dx++) {
                    cmd.text = strdup(txt);
                    cmd.x = px + dx;
                    cmd.y = py + dy;
                    add_cmd(pargs, &cmd);
                }
            }
        }
    } else
        logverb("No background behind text\n");

    cmd.layer = pargs->text_fg_layer;
    memcpy(cmd.rgba, pargs->rgba, sizeof(cmd.rgba));
    cmd.text = strdup(txt);
    cmd.x = px;
    cmd.y = py;
    add_cmd(pargs, &cmd);
}

void plotstuff_marker(plot_args_t* pargs, double x, double y) {
    cairo_t* cairo = pargs->cairo;
    cairo_move_to(cairo, x, y);
    cairoutils_draw_marker(cairo, pargs->marker, x, y, pargs->markersize);
}

int plotstuff_marker_radec(plot_args_t* pargs, double ra, double dec) {
    double x,y;
    //printf("plotstuff_marker_radec(%.3f, %.3f)\n", ra, dec);
    if (!plotstuff_radec2xy(pargs, ra, dec, &x, &y)) {
        ERROR("Failed to convert RA,Dec (%g,%g) to pixel position in plot_marker_radec\n", ra, dec);
        return -1;
    }
    assert(pargs->cairo);
    //logverb("plotstuff_marker_radec (%.3f, %.3f) -> (%.1f, %.1f)\n", ra, dec, x, y);
    // MAGIC 0.5 -- cairo/FITS coord offset
    plotstuff_marker(pargs, x-0.5, y-0.5);
    return 0;
}

int plotstuff_plot_stack(plot_args_t* pargs, cairo_t* cairo) {
    int i, j;
    int layer;
    anbool morelayers;

    logverb("Plotting %zu stacked plot commands.\n", bl_size(pargs->cairocmds));
    morelayers = TRUE;
    for (layer=0;; layer++) {
        if (!morelayers)
            break;
        morelayers = FALSE;
        for (i=0; i<bl_size(pargs->cairocmds); i++) {
            cairocmd_t* cmd = bl_access(pargs->cairocmds, i);
            if (cmd->layer > layer)
                morelayers = TRUE;
            if (cmd->layer != layer)
                continue;
            cairo_set_rgba(cairo, cmd->rgba);
            switch (cmd->type) {
            case CIRCLE:
                cairo_move_to(cairo, cmd->x + cmd->radius, cmd->y);
                cairo_arc(cairo, cmd->x, cmd->y, cmd->radius, 0, 2*M_PI);
                break;
            case MARKER:
                {
                    double oldmarkersize = pargs->markersize;
                    int oldmarker = pargs->marker;
                    pargs->markersize = cmd->markersize;
                    pargs->marker = cmd->marker;
                    plotstuff_marker(pargs, cmd->x, cmd->y);
                    pargs->markersize = oldmarkersize;
                    pargs->marker = oldmarker;
                }
                break;
            case TEXT:
                cairo_move_to(cairo, cmd->x, cmd->y);
                cairo_show_text(cairo, cmd->text);
                break;
            case LINE:
            case ARROW:
                plotstuff_move_to(pargs, cmd->x, cmd->y);
                plotstuff_line_to(pargs, cmd->x2, cmd->y2);
                {
                    double dx = cmd->x - cmd->x2;
                    double dy = cmd->y - cmd->y2;
                    double angle = atan2(dy, dx);
                    double dang = 30. * M_PI/180.0;
                    double arrowlen = 20;
                    plotstuff_line_to(pargs,
                                      cmd->x2 + cos(angle+dang)*arrowlen,
                                      cmd->y2 + sin(angle+dang)*arrowlen);
                    plotstuff_move_to(pargs, cmd->x2, cmd->y2);
                    plotstuff_line_to(pargs,
                                      cmd->x2 + cos(angle-dang)*arrowlen,
                                      cmd->y2 + sin(angle-dang)*arrowlen);
                }
                break;
            case RECTANGLE:
                cairo_move_to(cairo, cmd->x, cmd->y);
                cairo_line_to(cairo, cmd->x, cmd->y2);
                cairo_line_to(cairo, cmd->x2, cmd->y2);
                cairo_line_to(cairo, cmd->x2, cmd->y);
                cairo_close_path(cairo);
                if (cmd->fill)
                    cairo_fill(cairo);
                break;
            case POLYGON:
                if (!cmd->xy)
                    break;
                for (j=0; j<dl_size(cmd->xy)/2; j++)
                    (j == 0 ? cairo_move_to : cairo_line_to)(cairo, dl_get(cmd->xy, 2*j+0), dl_get(cmd->xy, 2*j+1));
                if (cmd->fill)
                    cairo_fill(cairo);
                break;
            }
            cairo_stroke(cairo);
        }
    }
    for (i=0; i<bl_size(pargs->cairocmds); i++) {
        cairocmd_t* cmd = bl_access(pargs->cairocmds, i);
        cairocmd_clear(cmd);
    }
    bl_remove_all(pargs->cairocmds);

    return 0;
}

static void plot_builtin_free(plot_args_t* pargs, void* baton) {
    anwcs_free(pargs->wcs);
    bl_free(pargs->cairocmds);
}

//static const plotter_t builtin = { "plot", plot_builtin_init, plot_builtin_init2, plot_builtin_command, plot_builtin_plot, plot_builtin_free, NULL };

DECLARE_PLOTTER(builtin) {
    DEFINE_PLOTTER_BODY(builtin)
        p->init2 = plot_builtin_init2;
    p->name = "plot";
}

int parse_image_format(const char* fmt) {
    if (strcaseeq(fmt, "png")) {
        return PLOTSTUFF_FORMAT_PNG;
    } else if (strcaseeq(fmt, "jpg") || strcaseeq(fmt, "jpeg")) {
        return PLOTSTUFF_FORMAT_JPG;
    } else if (strcaseeq(fmt, "ppm")) {
        return PLOTSTUFF_FORMAT_PPM;
    } else if (strcaseeq(fmt, "pdf")) {
        return PLOTSTUFF_FORMAT_PDF;
    } else if (strcaseeq(fmt, "fits") || strcaseeq(fmt, "fit")) {
        return PLOTSTUFF_FORMAT_FITS;
    }
    ERROR("Unknown image format \"%s\"", fmt);
    return -1;
}

int guess_image_format_from_filename(const char* fn) {
    // look for "."
    int len = strlen(fn);
    if (len >= 4 && fn[len-4] == '.') {
        return parse_image_format(fn + len - 3);
    }
    if (len >= 5 && fn[len - 5] == '.') {
        return parse_image_format(fn + len - 4);
    }
    return 0;
}

const char* image_format_name_from_code(int code) {
    if (code == PLOTSTUFF_FORMAT_JPG)
        return "jpeg";
    if (code == PLOTSTUFF_FORMAT_PNG)
        return "png";
    if (code == PLOTSTUFF_FORMAT_PPM)
        return "ppm";
    if (code == PLOTSTUFF_FORMAT_PDF)
        return "pdf";
    if (code == PLOTSTUFF_FORMAT_FITS)
        return "fits";
    if (code == PLOTSTUFF_FORMAT_MEMIMG)
        return "memory";
    return "unknown";
}

int plotstuff_set_color(plot_args_t* pargs, const char* name) {
    logverb("setting color to \"%s\"\n", name);
    return parse_color_rgba(name, pargs->rgba);
}

float plotstuff_get_alpha(const plot_args_t* pargs) {
    return pargs->rgba[3];
}

int plotstuff_set_alpha(plot_args_t* pargs, float alpha) {
    pargs->rgba[3] = alpha;
    return 0;
}

int plotstuff_set_bgcolor(plot_args_t* pargs, const char* name) {
    return parse_color_rgba(name, pargs->bg_rgba);
}

int plotstuff_set_bgrgba2(plot_args_t* pargs, float r, float g, float b, float a) {
    pargs->bg_rgba[0] = r;
    pargs->bg_rgba[1] = g;
    pargs->bg_rgba[2] = b;
    pargs->bg_rgba[3] = a;
    return 0;
}

int plotstuff_set_rgba(plot_args_t* pargs, const float* rgba) {
    pargs->rgba[0] = rgba[0];
    pargs->rgba[1] = rgba[1];
    pargs->rgba[2] = rgba[2];
    pargs->rgba[3] = rgba[3];
    return 0;
}

int plotstuff_set_rgba2(plot_args_t* pargs, float r, float g, float b, float a) {
    pargs->rgba[0] = r;
    pargs->rgba[1] = g;
    pargs->rgba[2] = b;
    pargs->rgba[3] = a;
    return 0;
}

int plotstuff_init(plot_args_t* pargs) {
    int i;

    memset(pargs, 0, sizeof(plot_args_t));

    /*
     plotters[0] = builtin;
     plotters[1] = plotter_fill;
     plotters[2] = plotter_xy;
     plotters[3] = plotter_image;
     plotters[4] = plotter_annotations;
     plotters[5] = plotter_grid;
     plotters[6] = plotter_outline;
     plotters[7] = plotter_index;
     plotters[8] = plotter_radec;
     plotters[9] = plotter_healpix;
     plotters[10] = plotter_match;
     */

    pargs->NP = 11;
    pargs->plotters = calloc(pargs->NP, sizeof(plotter_t));
    /*
     pargs->plotters[0] = builtin_new();
     pargs->plotters[1] = plot_fill_new();
     pargs->plotters[2] = plot_xy_new();
     pargs->plotters[3] = plot_image_new();
     pargs->plotters[4] = plot_annotations_new();
     pargs->plotters[5] = plot_grid_new();
     pargs->plotters[6] = plot_outline_new();
     pargs->plotters[7] = plot_index_new();
     pargs->plotters[8] = plot_radec_new();
     pargs->plotters[9] = plot_healpix_new();
     pargs->plotters[10] = plot_match_new();
     */
    //builtin_describe(pargs->plotters + 0);

    plot_builtin_describe    (pargs->plotters + 0);
    plot_fill_describe       (pargs->plotters + 1);
    plot_xy_describe         (pargs->plotters + 2);
    plot_image_describe      (pargs->plotters + 3);
    plot_annotations_describe(pargs->plotters + 4);
    plot_grid_describe       (pargs->plotters + 5);
    plot_outline_describe    (pargs->plotters + 6);
    plot_index_describe      (pargs->plotters + 7);
    plot_radec_describe      (pargs->plotters + 8);
    plot_healpix_describe    (pargs->plotters + 9);
    plot_match_describe      (pargs->plotters + 10);

    // First init
    for (i=0; i<pargs->NP; i++)
        pargs->plotters[i].baton = pargs->plotters[i].init(pargs);
    return 0;
}

int plotstuff_init2(plot_args_t* pargs) {
    int i;

    logverb("Creating drawing surface (%ix%i)\n", pargs->W, pargs->H);
    // Allocate cairo surface
    switch (pargs->outformat) {
    case PLOTSTUFF_FORMAT_PDF:
        if (pargs->outfn) {
            pargs->fout = fopen(pargs->outfn, "wb");
            if (!pargs->fout) {
                SYSERROR("Failed to open output file \"%s\"", pargs->outfn);
                return -1;
            }
        }
        pargs->target = cairo_pdf_surface_create_for_stream(cairoutils_file_write_func, pargs->fout, pargs->W, pargs->H);
        break;
    case PLOTSTUFF_FORMAT_JPG:
    case PLOTSTUFF_FORMAT_PPM:
    case PLOTSTUFF_FORMAT_PNG:
    case PLOTSTUFF_FORMAT_MEMIMG:
        pargs->target = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, pargs->W, pargs->H);
        if (!pargs->target) {
            ERROR("Failed to create Cairo image surface of size %i x %i\n", pargs->W, pargs->H);
            return -1;
        }
        if (cairo_surface_status(pargs->target) != CAIRO_STATUS_SUCCESS) {
            ERROR("Failed to create Cairo image surface of size %i x %i: %s\n", pargs->W, pargs->H,
                  cairo_status_to_string(cairo_surface_status(pargs->target)));
            return -1;
        }
        break;
    default:
        ERROR("Unknown output format %i", pargs->outformat);
        return -1;
        break;
    }
    pargs->cairo = cairo_create(pargs->target);

    /* D'oh, this flips the coord sys, but not text!
     // Flip the cairo reference frame (make 0,0 the bottom-left)
     cairo_scale(pargs->cairo, 1.0, -1.0);
     // FIXME -- could deal with 0.5 issues here!
     cairo_translate(pargs->cairo, 0.0, -pargs->H);
     */

    for (i=0; i<pargs->NP; i++) {
        if (pargs->plotters[i].init2 &&
            pargs->plotters[i].init2(pargs, pargs->plotters[i].baton)) {
            ERROR("Plot initializer failed");
            exit(-1);
        }
    }

    return 0;
}

void* plotstuff_get_config(plot_args_t* pargs, const char* name) {
    int i;
    for (i=0; i<pargs->NP; i++) {
        if (streq(pargs->plotters[i].name, name))
            return pargs->plotters[i].baton;
    }
    return NULL;
}

double plotstuff_pixel_scale(plot_args_t* pargs) {
    if (!pargs->wcs) {
        ERROR("plotstuff_pixel_scale: No WCS defined!");
        return 0.0;
    }
    return anwcs_pixel_scale(pargs->wcs);
}

anbool plotstuff_radec2xy(plot_args_t* pargs, double ra, double dec,
                          double* x, double* y) {
    if (!pargs->wcs) {
        ERROR("No WCS defined!");
        return FALSE;
    }
    return (anwcs_radec2pixelxy(pargs->wcs, ra, dec, x, y) ? FALSE : TRUE);
}

anbool plotstuff_xy2radec(plot_args_t* pargs, double x, double y,
                          double* pra, double* pdec) {
    if (!pargs->wcs) {
        ERROR("No WCS defined!");
        return FALSE;
    }
    return (anwcs_pixelxy2radec(pargs->wcs, x, y, pra, pdec)
            ? FALSE : TRUE);
}

anbool plotstuff_radec_is_inside_image(plot_args_t* pargs, double ra, double dec) {
    if (!pargs->wcs) {
        ERROR("No WCS defined!");
        return FALSE;
    }
    return anwcs_radec_is_inside_image(pargs->wcs, ra, dec);
}

void plotstuff_get_radec_bounds(const plot_args_t* pargs, int stepsize,
                                double* pramin, double* pramax,
                                double* pdecmin, double* pdecmax) {
    if (!pargs->wcs) {
        ERROR("No WCS defined!");
        return;
    }
    return anwcs_get_radec_bounds(pargs->wcs, stepsize, pramin, pramax, pdecmin, pdecmax);
}

int
ATTRIB_FORMAT(printf,2,3)
    plotstuff_run_commandf(plot_args_t* pargs, const char* format, ...) {
    char* str;
    va_list va;
    int rtn;
    va_start(va, format);
    if (vasprintf(&str, format, va) == -1) {
        ERROR("Failed to allocate temporary string to hold command");
        return -1;
    }
    rtn = plotstuff_run_command(pargs, str);
    va_end(va);
    return rtn;
}

int plotstuff_plot_layer(plot_args_t* pargs, const char* layer) {
    int i;
    for (i=0; i<pargs->NP; i++) {
        if (streq(layer, pargs->plotters[i].name)) {
            if (!pargs->cairo) {
                if (plotstuff_init2(pargs)) {
                    return -1;
                }
            }
            if (pargs->plotters[i].doplot) {
                if (pargs->plotters[i].doplot(layer, pargs->cairo, pargs, pargs->plotters[i].baton)) {
                    ERROR("Plotter \"%s\" failed on command \"%s\"", pargs->plotters[i].name, layer);
                    return -1;
                } else
                    return 0;
            }
        }
    }
    return -1;
}

int plotstuff_run_command(plot_args_t* pargs, const char* cmd) {
    int i;
    anbool matched = FALSE;
    if (!cmd || (strlen(cmd) == 0) || (cmd[0] == '#')) {
        return 0;
    }
    if (!plotstuff_plot_layer(pargs, cmd)) {
        return 0;
    }
    for (i=0; i<pargs->NP; i++) {
        if (starts_with(cmd, pargs->plotters[i].name)) {
            char* cmdcmd;
            char* cmdargs;
            if (!split_string_once(cmd, " ", &cmdcmd, &cmdargs)) {
                //ERROR("Failed to split command \"%s\" into words\n", cmd);
                //return -1;
                cmdcmd = strdup(cmd);
                cmdargs = NULL;
            }
            logmsg("Command \"%s\", args \"%s\"\n", cmdcmd, cmdargs);
            if (pargs->plotters[i].command(cmdcmd, cmdargs, pargs, pargs->plotters[i].baton)) {
                ERROR("Plotter \"%s\" failed on command \"%s\"", pargs->plotters[i].name, cmd);
                return -1;
            }
            free(cmdcmd);
            free(cmdargs);
        } else
            continue;
        matched = TRUE;
        break;
    }
    if (!matched) {
        ERROR("Did not find a plotter for command \"%s\"", cmd);
        return -1;
    }
    return 0;
}

int plotstuff_read_and_run_command(plot_args_t* pargs, FILE* f) {
    char* cmd;
    int rtn;
    cmd = read_string_terminated(stdin, "\n\r\0", 3, FALSE);
    logverb("command: \"%s\"\n", cmd);
    if (!cmd || feof(f)) {
        free(cmd);
        return -1;
    }
    rtn = plotstuff_run_command(pargs, cmd);
    free(cmd);
    return rtn;
}

int plotstuff_output(plot_args_t* pargs) {
    switch (pargs->outformat) {
    case PLOTSTUFF_FORMAT_PDF:

        if (pargs->outfn && !pargs->fout) {
            // open output file if it hasn't already been opened...
            pargs->fout = fopen(pargs->outfn, "wb");
            if (!pargs->fout) {
                SYSERROR("Failed to open output file \"%s\"", pargs->outfn);
                return -1;
            }
        }
        cairo_surface_flush(pargs->target);
        cairo_surface_finish(pargs->target);
        cairoutils_surface_status_errors(pargs->target);
        cairoutils_cairo_status_errors(pargs->cairo);
        if (pargs->outfn) {
            if (fclose(pargs->fout)) {
                SYSERROR("Failed to close output file \"%s\"", pargs->outfn);
                return -1;
            }
            pargs->fout = NULL;
        }
        break;

    case PLOTSTUFF_FORMAT_JPG:
    case PLOTSTUFF_FORMAT_PPM:
    case PLOTSTUFF_FORMAT_PNG:
    case PLOTSTUFF_FORMAT_MEMIMG:
        {
            int res;
            unsigned char* img = cairo_image_surface_get_data(pargs->target);
            // Convert image for output...
            cairoutils_argb32_to_rgba(img, pargs->W, pargs->H);
            if (pargs->outformat == PLOTSTUFF_FORMAT_MEMIMG) {
                pargs->outimage = img;
                res = 0;
                img = NULL;
            } else if (pargs->outformat == PLOTSTUFF_FORMAT_JPG) {
                res = cairoutils_write_jpeg(pargs->outfn, img, pargs->W, pargs->H);
            } else if (pargs->outformat == PLOTSTUFF_FORMAT_PPM) {
                res = cairoutils_write_ppm(pargs->outfn, img, pargs->W, pargs->H);
            } else if (pargs->outformat == PLOTSTUFF_FORMAT_PNG) {
                res = cairoutils_write_png(pargs->outfn, img, pargs->W, pargs->H);
            } else {
                res=-1; // for gcc
                assert(0);
            }
            if (res)
                ERROR("Failed to write output image");
            if (img)
                // Convert image back...
                cairoutils_rgba_to_argb32(img, pargs->W, pargs->H);
            return res;
        }
        break;
    default:
        ERROR("Unknown output format.");
        return -1;
    }
    return 0;
}

void plotstuff_get_maximum_rgba(plot_args_t* pargs,
                                int* p_r, int* p_g, int* p_b, int* p_a) {
    int i, r, g, b, a;
    uint32_t* ipix = (uint32_t*)cairo_image_surface_get_data(pargs->target);
    r = g = b = a = 0;
    for (i=0; i<(pargs->W * pargs->H); i++) {
        a = MAX(a, (ipix[i] >> 24) & 0xff);
        r = MAX(r, (ipix[i] >> 16) & 0xff);
        g = MAX(g, (ipix[i] >>  8) & 0xff);
        b = MAX(b, (ipix[i]      ) & 0xff);
    }
    if (p_r)
        *p_r = r;
    if (p_g)
        *p_g = g;
    if (p_b)
        *p_b = b;
    if (p_a)
        *p_a = a;
}
	

void plotstuff_free(plot_args_t* pargs) {
    int i;
    for (i=0; i<pargs->NP; i++) {
        pargs->plotters[i].free(pargs, pargs->plotters[i].baton);
    }
    cairo_destroy(pargs->cairo);
    cairo_surface_destroy(pargs->target);
}

