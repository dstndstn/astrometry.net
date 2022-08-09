/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef ANWCSLIB_H
#define ANWCSLIB_H

#include "astrometry/sip.h"
#include "astrometry/an-bool.h"
#include "astrometry/qfits_header.h"
#include "astrometry/bl.h"

/** Interface to Mark Calabretta's wcslib, if available, and
 Astrometry.net's TAN/SIP implementation.  Also WCSTools. */

#define ANWCS_TYPE_WCSLIB 1
#define ANWCS_TYPE_SIP 2
#define ANWCS_TYPE_WCSTOOLS 3

struct anwcs_t {
    /**
     If type == ANWCS_TYPE_WCSLIB:
     data is a private struct containing a wcslib  "struct wcsprm*".

     If type == ANWCS_TYPE_SIP:
     data is a "sip_t*"

     If type == ANWCS_TYPE_WCSTOOLS:
     data is a "struct WorldCoor*"
     */
    int type;
    void* data;
};
typedef struct anwcs_t anwcs_t;



pl* anwcs_walk_outline(const anwcs_t* wcs, const dl* path, int fill);



// len: length in characters of 'str'
anwcs_t* anwcs_wcslib_from_string(const char* str, int len);
char* anwcs_wcslib_to_string(const anwcs_t* wcs, char** s, int* len);

anwcs_t* anwcs_open(const char* filename, int ext);

anwcs_t* anwcs_open_wcslib(const char* filename, int ext);

anwcs_t* anwcs_open_wcstools(const char* filename, int ext);

anwcs_t* anwcs_wcstools_from_string(const char* str, int len);

anwcs_t* anwcs_open_sip(const char* filename, int ext);

anwcs_t* anwcs_open_tan(const char* filename, int ext);

anwcs_t* anwcs_new_sip(const sip_t* sip);

anwcs_t* anwcs_new_tan(const tan_t* tan);

// Creates an axis-aligned TAN WCS at the given RA,Dec with "width" width in degrees
// and W x H  pixels.
anwcs_t* anwcs_create_box(double ra, double dec, double width, int W, int H);

anwcs_t* anwcs_create_box_upsidedown(double ra, double dec, double width, int W, int H);

anwcs_t* anwcs_create_mercator(double refra, double refdec,
                               double zoomfactor,
                               int W, int H, anbool yflip);

anwcs_t* anwcs_create_mercator_2(double refra, double refdec,
                                 double crpix1, double crpix2,
                                 double zoomfactor,
                                 int W, int H, anbool yflip);

anwcs_t* anwcs_create_mollweide(double refra, double refdec,
                                double zoomfactor,
                                int W, int H, anbool yflip);

anwcs_t* anwcs_create_cea_wcs(double refra, double refdec,
                              double refx, double refy,
                              double pixscale,
                              int W, int H, anbool yflip);

// In-place conversion from GLON,GLAT to RA,DEC coordinates.  Requires WCSLIB >= 7.5
int anwcs_galactic_to_radec(anwcs_t* wcs);

/* plate CARree projection in Galactic coordinates. */
anwcs_t* anwcs_create_galactic_car_wcs(double refra, double refdec,
                                       double refx, double refy,
                                       double pixscale,
                                       int W, int H, anbool yflip);

anwcs_t* anwcs_create_hammer_aitoff(double refra, double refdec,
                                    double zoomfactor,
                                    int W, int H, anbool yflip);

// Sets the pixel scale based on the width and zoom factor (same pixel scale horiz. & vert.)
anwcs_t* anwcs_create_hammer_aitoff_rectangular(double refra, double refdec,
                                                double zoomfactor, double rotate,
                                                int W, int H, anbool yflip);

anwcs_t* anwcs_create_hammer_aitoff_galactic(double ref_long, double ref_lat,
                                             double zoomfactor,
                                             int W, int H, anbool yflip);

anwcs_t* anwcs_create_allsky_hammer_aitoff(double refra, double refdec,
                                           int W, int H);
anwcs_t* anwcs_create_allsky_hammer_aitoff2(double refra, double refdec,
                                            int W, int H);

int anwcs_write(const anwcs_t* wcs, const char* filename);

int anwcs_write_to(const anwcs_t* wcs, FILE* fid);

int anwcs_add_to_header(const anwcs_t* wcs, qfits_header* hdr);

int anwcs_radec2pixelxy(const anwcs_t* wcs, double ra, double dec, double* p_x, double* p_y);

int anwcs_pixelxy2radec(const anwcs_t* wcs, double px, double py, double* ra, double* dec);

int anwcs_pixelxy2xyz(const anwcs_t* wcs, double px, double py, double* p_xyz);

int anwcs_xyz2pixelxy(const anwcs_t* wcs, const double* xyz, double *px, double *py);

anbool anwcs_radec_is_inside_image(const anwcs_t* wcs, double ra, double dec);

void anwcs_get_cd_matrix(const anwcs_t* wcs, double* p_cd);

/**
 The SIP implementation guarantees:

 ramin <= ramax
 ramin may be < 0, or ramax > 360, if the image straddles RA=0.
 */
void anwcs_get_radec_bounds(const anwcs_t* wcs, int stepsize,
                            double* pramin, double* pramax,
                            double* pdecmin, double* pdecmax);

void anwcs_print(const anwcs_t* wcs, FILE* fid);

// useful for python
void anwcs_print_stdout(const anwcs_t* wcs);

// Center and radius of the field.
// RA,Dec,radius in degrees.
int anwcs_get_radec_center_and_radius(const anwcs_t* anwcs,
                                      double* p_ra, double* p_dec, double* p_radius);

void anwcs_walk_image_boundary(const anwcs_t* wcs, double stepsize,
                               void (*callback)(const anwcs_t* wcs, double x, double y, double ra, double dec, void* token),
                               void* token);

anbool anwcs_find_discontinuity(const anwcs_t* wcs, double ra1, double dec1,
                                double ra2, double dec2,
                                double* pra3, double* pdec3,
                                double* pra4, double* pdec4);

anbool anwcs_is_discontinuous(const anwcs_t* wcs, double ra1, double dec1,
                              double ra2, double dec2);

/*
 // Assuming there is a discontinuity between (ra1,dec1) and (ra2,dec2),
 // return 
 int anwcs_get_discontinuity(const anwcs_t* wcs, double ra1, double dec1,
 double ra2, double dec2,
 double* dra, double* ddec);
 */
dl* anwcs_walk_discontinuity(const anwcs_t* wcs,
                             double ra1, double dec1, double ra2, double dec2,
                             double ra3, double dec3, double ra4, double dec4,
                             double stepsize,
                             dl* radecs);

anbool anwcs_overlaps(const anwcs_t* wcs1, const anwcs_t* wcs2, int stepsize);

double anwcs_imagew(const anwcs_t* anwcs);
double anwcs_imageh(const anwcs_t* anwcs);

void anwcs_set_size(anwcs_t* anwcs, int W, int H);

int anwcs_scale_wcs(anwcs_t* anwcs, double scale);

// angle in deg
int anwcs_rotate_wcs(anwcs_t* anwcs, double angle);

// Approximate pixel scale, in arcsec/pixel, at the reference point.
double anwcs_pixel_scale(const anwcs_t* anwcs);

void anwcs_free(anwcs_t* wcs);

// useful for python: get the sip_t*, if this anwcs wraps a SIP structure; NULL else
sip_t* anwcs_get_sip(const anwcs_t* wcs);

#endif
