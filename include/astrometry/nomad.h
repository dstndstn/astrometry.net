/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef NOMAD_H
#define NOMAD_H

#include <stdint.h>

#include "astrometry/starutil.h"

#define NOMAD_RECORD_SIZE 88

/*
 See: http://www.nofs.navy.mil/nomad/nomad_readme.html
 */

struct nomad_entry {
    // [degrees]
    double ra;
    double dec;

    // [degrees]
    float sigma_racosdec;
    float sigma_dec;

    // [arcsec/yr]
    float pm_racosdec;
    float pm_dec;

    // [arcsec/yr]
    float sigma_pm_racosdec;
    float sigma_pm_dec;

    // [yr]
    float epoch_ra;
    float epoch_dec;

    // mag
    // (30.000 in any magnitude field indicates "no data".)
    // (30.001 = "no data" but ...)
    float mag_B;
    float mag_V;
    float mag_R;
    float mag_J;
    float mag_H;
    float mag_K;

    int32_t usnob_id;
    int32_t twomass_id;
    int32_t yb6_id;
    int32_t ucac2_id;
    int32_t tycho2_id;

    // all these take values from the "nomad_src" enum.
    uint8_t astrometry_src;
    uint8_t blue_src;
    uint8_t visual_src;
    uint8_t red_src;

    anbool usnob_fail;       // UBBIT   "Fails Blaise's test for USNO-B1.0 star"
    anbool twomass_fail;     // TMBIT   "Fails Roc's test for clean 2MASS star"
    anbool tycho_astrometry; // TYBIT   "Astrometry comes from Tycho2"
    anbool alt_radec;        // XRBIT   "Alt correlations for same (RA,Dec)"
    // This bit is NEVER set in NOMAD.
    //anbool alt_2mass;        // ITMBIT  "Alt correlations for same 2MASS ID"
    anbool alt_ucac;         // IUCBIT  "Alt correlations for same UCAC-2 ID"
    anbool alt_tycho;        // ITYBIT  "Alt correlations for same Tycho2 ID"
    anbool blue_o;           // OMAGBIT "Blue magnitude from O (not J) plate"
    anbool red_e;            // EMAGBIT "Red magnitude from E (not F) plate"
    anbool twomass_only;     // TMONLY  "Object found only in 2MASS cat"
    anbool hipp_astrometry;  // HIPAST  "Ast from Hipparcos (not Tycho2) cat"
    anbool diffraction;      // SPIKE   "USNO-B1.0 diffraction spike bit set"
    anbool confusion;        // TYCONF  "Tycho2 confusion flag"
    anbool bright_confusion; // BSCONF  "Bright star has nearby faint source"
    anbool bright_artifact;  // BSART   "Faint source is bright star artifact"
    anbool standard;         // USEME   "Recommended astrometric standard"
    // This bit is NEVER set in NOMAD.
    //anbool external;         // EXCAT   "External, non-astrometric object"

    // this is a staging area for FITS i/o.
    uint8_t flags[2];

    // sequence number assigned by us (it's not in the original catalogue),
    // composed of the 1/10 degree DEC zone (top 11 bits) and the sequence
    // number within the zone (bottom 21 bits).
    uint32_t nomad_id;
};
typedef struct nomad_entry nomad_entry;

enum nomad_src {
    NOMAD_SRC_NONE = 0,
    NOMAD_SRC_USNOB,
    NOMAD_SRC_2MASS,
    NOMAD_SRC_YB6,
    NOMAD_SRC_UCAC2,
    NOMAD_SRC_TYCHO2,
    NOMAD_SRC_HIPPARCOS,
};

int nomad_parse_entry(nomad_entry* entry, const void* encoded);

#endif

