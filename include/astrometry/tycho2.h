/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

/*
 Parses Tycho-2 data files, which are available here:
 http://www.astro.ku.dk/~cf/CD/data/
 */

#ifndef TYCHO2_H
#define TYCHO2_H

#include <stdint.h>

#include "astrometry/starutil.h"

// 206 bytes of data, but each record is supposed to be terminated
// by \r\n, making...
#define TYCHO_RECORD_SIZE_RAW 206

// ... 208 bytes total!
#define TYCHO_RECORD_SIZE 208

#define TYCHO_SUPPLEMENT_RECORD_SIZE_RAW 122
#define TYCHO_SUPPLEMENT_RECORD_SIZE 124

struct tycho2_entry {
    // together these form the star ID.
    int16_t tyc1;  // [1, 9537] in main catalog; [2, 9529] in suppl.
    int16_t tyc2;  // [1, 12121] main; [1,12112] suppl.
    uint8_t tyc3;  // [1, 3] main, [1, 4] suppl.

    // flag "P": photo-center of two stars was used for position
    anbool photo_center;
    // flag "X": no mean position, no proper motion
    anbool no_motion;
    // flag "T"
    anbool tycho1_star;
    // flag "D"
    anbool double_star;
    // flag "P" ( DP)
    anbool photo_center_treatment;
    // flag "H" (in supplements)
    anbool hipparcos_star;

    // [degrees]
    double ra;         // RAdeg
    double dec;        // DEdeg
    double mean_ra;     // mRAdeg
    double mean_dec;    // mDEdeg

    // [degrees]
    float sigma_ra;    // mas2deg(e_RA*)
    float sigma_dec;   // mas2deg(e_DE)
    float sigma_mean_ra;   // mas2deg(e_mRA*)
    float sigma_mean_dec;  // mas2deg(e_mDE)

    // [arcsec/yr]
    float pm_ra;        // pmRA* / 1000
    // (note that this is actually a change in RA*cos(Dec))
    float pm_dec;       // pmDE / 1000

    // [arcsec/yr]
    float sigma_pm_ra;  // e_pmRA* / 1000
    float sigma_pm_dec; // e_pmDE / 1000

    // [yr]
    float epoch_ra;    // epRA + 1990
    float epoch_dec;   // epDE + 1990
    float epoch_mean_ra;   // mepRA
    float epoch_mean_dec;  // mepDE
    //
    uint8_t nobs; // Num

    // "goodness"
    float goodness_mean_ra;  // g_mRA
    float goodness_mean_dec; // g_mDEC
    float goodness_pm_ra;  // g_pmRA
    float goodness_pm_dec; // g_pmDEC

    // [mag] (0.0 means unavailable)
    float mag_BT;        // BT
    float mag_VT;        // VT
    float sigma_BT;      // e_BT
    float sigma_VT;      // e_VT

    // [mag] supplements only: Hp magnitude
    float mag_HP;        // HP
    float sigma_HP;      // e_HP

    // [degrees], -1.0 for null.
    float prox;

    float correlation;

    int32_t hipparcos_id;    // [1, 120404] (or zero for null)
    char hip_ccdm[4];     // (up to three chars; null-terminated.)
};
typedef struct tycho2_entry tycho2_entry;

int tycho2_guess_is_supplement(const char* line);

int tycho2_parse_entry(const char* line, tycho2_entry* entry);

int tycho2_supplement_parse_entry(const char* line, tycho2_entry* entry);

#endif

