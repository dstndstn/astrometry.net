/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef TWOMASS_H
#define TWOMASS_H

#include <stdint.h>

#include "astrometry/an-bool.h"

/**

 See:
 .    ftp://ftp.ipac.caltech.edu/pub/2mass/allsky/format_psc.html

 */
struct twomass_entry {
    // [degrees] - J2000 ICRS
    double ra;
    // [degrees] - J2000 ICRS
    double dec;

    // unique id of this object.
    unsigned int key;

    // [degrees] - one-sigma positional error ellipse: major axis
    float err_major;
    // [degrees] - one-sigma positional error ellipse: minor axis
    float err_minor;
    // [degrees] - rotation east of north of the major axis of the error ellipse
    float err_angle;

    // hhmmssss[+-]ddmmsss[ABC...]
    char designation[18];

    // is this in the northern hemisphere?
    // TRUE=northern, FALSE=southern.
    anbool northern_hemisphere;

    // [mag] - J magnitude
    float j_m;
    // [mag] - J corrected photometric uncertainty
    float j_cmsig;
    // [mag] - J total photometric uncertainty
    float j_msigcom;
    // J signal-to-noise
    float j_snr;

    // [mag] - H magnitude
    float h_m;
    // [mag] - H corrected photometric uncertainty
    float h_cmsig;
    // [mag] - H total photometric uncertainty
    float h_msigcom;
    // H signal-to-noise
    float h_snr;

    // [mag] - K magnitude
    float k_m;
    // [mag] - K corrected photometric uncertainty
    float k_cmsig;
    // [mag] - K total photometric uncertainty
    float k_msigcom;
    // K signal-to-noise
    float k_snr;

    // [code: twomass_quality_val] photometric quality flags
    char j_quality;
    char h_quality;
    char k_quality;

    // read flags: where did the data come from?
    uint8_t j_read_flag;
    uint8_t h_read_flag;
    uint8_t k_read_flag;

    // blend flags: how many source measurements were blended?
    uint8_t h_blend_flag;
    uint8_t j_blend_flag;
    uint8_t k_blend_flag;

    // [code: twomass_cc_val] contamination and confusion flags: object detection.
    char j_cc;
    char h_cc;
    char k_cc;

    // number of detections (M=seen, N=possible)
    uint8_t j_ndet_M;
    uint8_t j_ndet_N;
    uint8_t h_ndet_M;
    uint8_t h_ndet_N;
    uint8_t k_ndet_M;
    uint8_t k_ndet_N;

    // may be a foreground star superimposed on a galaxy.
    uint8_t galaxy_contam;

    // [degrees] - proximity to the nearest other source in the catalog.
    float proximity;
    // [degrees] - angle east of north to the nearest object
    float prox_angle;
    // key of the nearest neighbour.
    unsigned int prox_key;

    // day the observation run was started
    uint16_t date_year;
    uint8_t date_month;
    uint8_t date_day;

    // [days]: Julian date (+- 30 seconds) of the measurement.
    double jdate;

    // nightly scan number
    uint16_t scan;
    // may be a minor planet, comet, asteroid, etc.
    anbool minor_planet;

    // [degrees] - angle east of north to optical counterpart
    float phi_opt;

    // [degrees] - galactic longitude
    float glon;
    // [degrees] - galactic latitude
    float glat;

    // [degrees]
    float x_scan;

    // (null)
    float j_psfchi;
    // [mag] (null)
    float j_m_stdap;
    // [mag] (null)
    float j_msig_stdap;

    // (null)
    float h_psfchi;
    // [mag] (null)
    float h_m_stdap;
    // [mag] (null)
    float h_msig_stdap;

    // (null)
    float k_psfchi;
    // [mag] (null)
    float k_m_stdap;
    // [mag] (null)
    float k_msig_stdap;

    // [degrees] (null):
    float dist_opt;
    // [mag] (null)
    float b_m_opt;
    // [mag] (null)
    float vr_m_opt;

    // [degrees]
    float dist_edge_ns;
    // [degrees]
    float dist_edge_ew;
    // TRUE=north
    anbool dist_flag_ns;
    // TRUE=east
    anbool dist_flag_ew;

    uint8_t dup_src;
    anbool use_src;

    // [code: twomass_association_val]
    char association;

    uint8_t nopt_mchs;

    uint16_t coadd;

    unsigned int scan_key;

    // AKA ext_key (null)
    unsigned int xsc_key;

    unsigned int coadd_key;
};
typedef struct twomass_entry twomass_entry;

#define TWOMASS_NULL (1.0/0.0)

#define TWOMASS_KEY_NULL 0xffffff

enum twomass_association_val {
    TWOMASS_ASSOCIATION_NONE,
    TWOMASS_ASSOCIATION_TYCHO,
    TWOMASS_ASSOCIATION_USNOA2
};

enum twomass_quality_val {
    TWOMASS_QUALITY_NO_BRIGHTNESS,    // X flag
    TWOMASS_QUALITY_UPPER_LIMIT_MAG,  // U
    TWOMASS_QUALITY_NO_SIGMA,         // F
    TWOMASS_QUALITY_BAD_FIT,          // E
    TWOMASS_QUALITY_A,
    TWOMASS_QUALITY_B,
    TWOMASS_QUALITY_C,
    TWOMASS_QUALITY_D
};

enum twomass_cc_val {
    TWOMASS_CC_NONE,            // 0 flag
    TWOMASS_CC_PERSISTENCE,     // p flag
    TWOMASS_CC_CONFUSION,       // c
    TWOMASS_CC_DIFFRACTION,     // d
    TWOMASS_CC_STRIPE,          // s
    TWOMASS_CC_BANDMERGE        // b
};

int twomass_is_value_null(float val);

int twomass_parse_entry(twomass_entry* entry, const char* line);

int twomass_cc_flag(unsigned char val, unsigned char flag);

int twomass_quality_flag(unsigned char val, unsigned char flag);

int twomass_is_null_float(float f);

#endif
