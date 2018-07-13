/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "os-features.h"
#include "2mass.h"
#include "starutil.h"

int twomass_is_value_null(float val) {
    return (!isfinite(val));
}

// parses a value that may be null (represented by "\N" in the 2MASS data files).
// expects the value to be followed by "|".
static int parse_null(const char** pcursor, float* dest) {
    int nchars;
    if (!strncmp(*pcursor, "\\N|", 3)) {
        *dest = TWOMASS_NULL;
        *pcursor += 3;
        return 0;
    }
    if (sscanf(*pcursor, "%f|%n", dest, &nchars) != 1) {
        return -1;
    }
    *pcursor += nchars;
    return 0;
}

static int parse_quality_flag(char flag, char* val) {
    switch (flag) {
    case 'X':
        *val = TWOMASS_QUALITY_NO_BRIGHTNESS;
        break;
    case 'U':
        *val = TWOMASS_QUALITY_UPPER_LIMIT_MAG;
        break;
    case 'F':
        *val = TWOMASS_QUALITY_NO_SIGMA;
        break;
    case 'E':
        *val = TWOMASS_QUALITY_BAD_FIT;
        break;
    case 'A':
        *val = TWOMASS_QUALITY_A;
        break;
    case 'B':
        *val = TWOMASS_QUALITY_B;
        break;
    case 'C':
        *val = TWOMASS_QUALITY_C;
        break;
    case 'D':
        *val = TWOMASS_QUALITY_D;
        break;
    default:
        return -1;
    }
    return 0;
}

static int parse_cc_flag(char flag, char* val) {
    switch (flag) {
    case 'p':
        *val = TWOMASS_CC_PERSISTENCE;
        break;
    case 'c':
        *val = TWOMASS_CC_CONFUSION;
        break;
    case 'd':
        *val = TWOMASS_CC_DIFFRACTION;
        break;
    case 's':
        *val = TWOMASS_CC_STRIPE;
        break;
    case 'b':
        *val = TWOMASS_CC_BANDMERGE;
        break;
    case '0':
        *val = TWOMASS_CC_NONE;
        break;
    default:
        return -1;
    }
    return 0;
}


#define ensure(c, f) { if (c!='|') { fprintf(stderr, "Expected '|' following field %s in 2MASS line.\n", f); return -1; }}

//#define printval printf
//#define printval(s,args...) printf(s,args)
#define printval(s,args...) {}

int twomass_cc_flag(unsigned char val, unsigned char flag) {
    return (val == flag);
}

int twomass_quality_flag(unsigned char val, unsigned char flag) {
    return (val == flag);
}

int twomass_is_null_float(float f) {
    return isnan(f);
}

int twomass_parse_entry(struct twomass_entry* e, const char* line) {
    const char* cursor;
    int nchars;
    int i;
    double vals1[5];
    float val2;
    float val3;

    cursor = line;
    for (i=0; i<5; i++) {
        char* names[] = { "ra", "dec", "err_maj", "err_min", "err_ang" };
        if (sscanf(cursor, "%lf|%n", vals1+i, &nchars) != 1) {
            fprintf(stderr, "Failed to parse \"%s\" entry in 2MASS line.\n", names[i]);
            return -1;
        }
        cursor += nchars;
    }

    // note: the bracketed units are the units used in the [source file : struct];
    // if only unit is given then it's the same in both.

    e->ra = vals1[0];  // [deg]
    e->dec = vals1[1]; // [deg]
    e->err_major = arcsec2deg(vals1[2]); // [arcsec : deg]
    e->err_minor = arcsec2deg(vals1[3]); // [arcsec : deg]
    e->err_angle = vals1[4]; // [deg]

    printval("ra %g, dec %g, err_major %g, err_minor %g, err_angle %i\n", e->ra, e->dec, e->err_major, e->err_minor, e->err_angle);

    strncpy(e->designation, cursor, 17);
    e->designation[17] = '\0';
    if (strlen(e->designation) != 17) {
        fprintf(stderr, "Failed to parse \"designation\" entry in 2MASS line.\n");
        return -1;
    }
    cursor += 18;

    printval("designation %s\n", e->designation);

    for (i=0; i<12; i++) {
        char* names[] = {
            "j_m", "j_cmsig", "j_msigcom", "j_snr",
            "h_m", "h_cmsig", "h_msigcom", "h_snr",
            "k_m", "k_cmsig", "k_msigcom", "k_snr" };
        float* dests[] = {
            &e->j_m, &e->j_cmsig, &e->j_msigcom, &e->j_snr,
            &e->h_m, &e->h_cmsig, &e->h_msigcom, &e->h_snr,
            &e->k_m, &e->k_cmsig, &e->k_msigcom, &e->k_snr };
        // these are all [mag] or unitless
        if (parse_null(&cursor, dests[i])) {
            fprintf(stderr, "Failed to parse \"%s\" entry in 2MASS line.\n", names[i]);
            return -1;
        }
    }

    printval("j_m %g, j_cmsig %g, j_msigcom %g, j_snr %g.\n", e->j_m, e->j_cmsig, e->j_msigcom, e->j_snr);
    printval("h_m %g, h_cmsig %g, h_msigcom %g, h_snr %g.\n", e->h_m, e->h_cmsig, e->h_msigcom, e->h_snr);
    printval("k_m %g, k_cmsig %g, k_msigcom %g, k_snr %g.\n", e->k_m, e->k_cmsig, e->k_msigcom, e->k_snr);

    for (i=0; i<3; i++) {
        char bands[] = { 'j', 'h', 'k' };
        char* quals[] = { &e->j_quality, &e->h_quality, &e->k_quality };
        if (parse_quality_flag(*cursor, quals[i])) {
            fprintf(stderr, "Failed to parse '%c_quality' entry in 2MASS line.\n", bands[i]);
            return -1;
        }
        cursor++;
    }

    printval("j X=%i, U=%i, F=%i, E=%i, A=%i, B=%i, C=%i, D=%i\n",
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_NO_BRIGHTNESS),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_UPPER_LIMIT_MAG),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_NO_SIGMA),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_BAD_FIT),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_A),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_B),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_C),
             twomass_quality_flag(e->j_quality, TWOMASS_QUALITY_D));

    printval("h X=%i, U=%i, F=%i, E=%i, A=%i, B=%i, C=%i, D=%i\n",
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_NO_BRIGHTNESS),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_UPPER_LIMIT_MAG),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_NO_SIGMA),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_BAD_FIT),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_A),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_B),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_C),
             twomass_quality_flag(e->h_quality, TWOMASS_QUALITY_D));

    printval("k X=%i, U=%i, F=%i, E=%i, A=%i, B=%i, C=%i, D=%i\n",
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_NO_BRIGHTNESS),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_UPPER_LIMIT_MAG),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_NO_SIGMA),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_BAD_FIT),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_A),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_B),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_C),
             twomass_quality_flag(e->k_quality, TWOMASS_QUALITY_D));

    ensure(*cursor, "flags");
    cursor++;

    for (i=0; i<3; i++) {
        uint8_t* dests[] = { &e->j_read_flag, &e->h_read_flag, &e->k_read_flag };

        if (*cursor < '0' || *cursor > '9') {
            fprintf(stderr, "Error parsing read_flag from 2MASS entry.\n");
            return -1;
        }
        *(dests[i]) = *cursor - '0';
        cursor++;
    }
    printval("read_flag j=%i, h=%i, k=%i.\n", e->j_read_flag, e->h_read_flag, e->k_read_flag);

    ensure(*cursor, "read_flag");
    cursor++;

    for (i=0; i<3; i++) {
        uint8_t* dests[] = { &e->j_blend_flag, &e->h_blend_flag, &e->k_blend_flag };
        if (*cursor < '0' || *cursor > '9') {
            fprintf(stderr, "Failed to parse blend_flag field in a 2MASS entry.\n");
            return -1;
        }
        *(dests[i]) = *cursor - '0';
        cursor++;
    }
    printval("blend_flag j=%i, h=%i, k=%i.\n", e->j_blend_flag, e->h_blend_flag, e->k_blend_flag);

    ensure(*cursor, "blend_flag");
    cursor++;

    for (i=0; i<3; i++) {
        char bands[] = { 'j', 'h', 'k' };
        char* ccs[] = { &e->j_cc, &e->h_cc, &e->k_cc };
        if (parse_cc_flag(*cursor, ccs[i])) {
            fprintf(stderr, "Failed to parse '%c_cc' entry in 2MASS line.\n", bands[i]);
            return -1;
        }
        cursor++;
    }

    printval("j_confusion: p=%i, c=%i, d=%i, s=%i, b=%i.\n",
             twomass_cc_flag(e->j_cc, TWOMASS_CC_PERSISTENCE),
             twomass_cc_flag(e->j_cc, TWOMASS_CC_CONFUSION),
             twomass_cc_flag(e->j_cc, TWOMASS_CC_DIFFFRACTION),
             twomass_cc_flag(e->j_cc, TWOMASS_CC_STRIPE),
             twomass_cc_flag(e->j_cc, TWOMASS_CC_BANDMERGE));
    printval("h_confusion: p=%i, c=%i, d=%i, s=%i, b=%i.\n",
             twomass_cc_flag(e->h_cc, TWOMASS_CC_PERSISTENCE),
             twomass_cc_flag(e->h_cc, TWOMASS_CC_CONFUSION),
             twomass_cc_flag(e->h_cc, TWOMASS_CC_DIFFFRACTION),
             twomass_cc_flag(e->h_cc, TWOMASS_CC_STRIPE),
             twomass_cc_flag(e->h_cc, TWOMASS_CC_BANDMERGE));
    printval("k_confusion: p=%i, c=%i, d=%i, s=%i, b=%i.\n",
             twomass_cc_flag(e->k_cc, TWOMASS_CC_PERSISTENCE),
             twomass_cc_flag(e->k_cc, TWOMASS_CC_CONFUSION),
             twomass_cc_flag(e->k_cc, TWOMASS_CC_DIFFFRACTION),
             twomass_cc_flag(e->k_cc, TWOMASS_CC_STRIPE),
             twomass_cc_flag(e->k_cc, TWOMASS_CC_BANDMERGE));

    ensure(*cursor, "confusion_flag");
    cursor++;

    assert(*cursor >= '0' && *cursor <= '9');
    e->j_ndet_M = *cursor - '0';
    cursor++;
    assert(*cursor >= '0' && *cursor <= '9');
    e->j_ndet_N = *cursor - '0';
    cursor++;
    assert(*cursor >= '0' && *cursor <= '9');
    e->h_ndet_M = *cursor - '0';
    cursor++;
    assert(*cursor >= '0' && *cursor <= '9');
    e->h_ndet_N = *cursor - '0';
    cursor++;
    assert(*cursor >= '0' && *cursor <= '9');
    e->k_ndet_M = *cursor - '0';
    cursor++;
    assert(*cursor >= '0' && *cursor <= '9');
    e->k_ndet_N = *cursor - '0';
    cursor++;

    printval("j_ndet %i/%i\n", e->j_ndet_M, e->j_ndet_N);
    printval("h_ndet %i/%i\n", e->h_ndet_M, e->h_ndet_N);
    printval("k_ndet %i/%i\n", e->k_ndet_M, e->k_ndet_N);

    ensure(*cursor, "ndet");
    cursor++;

    if (sscanf(cursor, "%f|%n", &e->proximity, &nchars) != 1) {
        fprintf(stderr, "Failed to parse 'proximity' entry in 2MASS line.\n");
        return -1;
    }
    cursor += nchars;
    e->proximity = arcsec2deg(e->proximity); // [arcsec : deg]
    printval("proximity %g\n", e->proximity);

    if (parse_null(&cursor, &val2)) {
        fprintf(stderr, "Failed to parse 'prox_angle' entry in 2MASS line.\n");
        return -1;
    }
    if (twomass_is_null_float(val2))
        e->prox_angle = TWOMASS_NULL;
    else
        e->prox_angle = val2; // [deg]

    printval("proximity_angle %i\n", e->prox_angle);

    if (sscanf(cursor, "%u|%n", &e->prox_key, &nchars) != 1) {
        fprintf(stderr, "Failed to parse 'prox_key' entry in 2MASS line.\n");
        return -1;
    }
    cursor += nchars;
    printval("proximity_key %i\n", e->prox_key);

    if (*cursor < '0' || *cursor > '2') {
        fprintf(stderr, "Failed to parse 'galaxy_contam' entry in 2MASS line.\n");
        return -1;
    }
    e->galaxy_contam = *cursor - '0';
    cursor ++;
    printval("galaxy contamination %i\n", e->galaxy_contam);

    ensure(*cursor, "galaxy_contam");
    cursor++;

    if (*cursor < '0' || *cursor > '1') {
        fprintf(stderr, "Failed to parse 'minor_planet' entry in 2MASS line.\n");
        return -1;
    }
    e->minor_planet = (anbool)(*cursor - '0');
    cursor++;
    printval("minor planet %i\n", e->minor_planet);

    ensure(*cursor, "minor_planet");
    cursor++;

    if (sscanf(cursor, "%u|%n", &e->key, &nchars) != 1) {
        fprintf(stderr, "Failed to parse 'key' entry in 2MASS line.\n");
        return -1;
    }
    cursor += nchars;
    printval("key %i\n", e->key);

    switch (*cursor) {
    case 'n':
        e->northern_hemisphere = TRUE;
        break;
    case 's':
        e->northern_hemisphere = FALSE;
        break;
    default:
        fprintf(stderr, "Failed to parse 'northern_hemisphere' entry in 2MASS line.\n");
        return -1;
    }
    cursor ++;
    printval("hemisphere %s\n", (e->northern_hemisphere ? "N" : "S"));

    ensure(*cursor, "northern_hemisphere");
    cursor++;

    {
        unsigned int yr, mon, day, scan;
        if (sscanf(cursor, "%u-%u-%u|%n", &yr, &mon, &day, &nchars) != 3) {
            fprintf(stderr, "Failed to parse 'date' entry in 2MASS line.\n");
            return -1;
        }
        cursor += nchars;
        e->date_year = yr;
        e->date_month = mon;
        e->date_day = day;

        if (sscanf(cursor, "%u|%n", &scan, &nchars) != 1) {
            fprintf(stderr, "Failed to parse 'scan' entry in 2MASS line.\n");
            return -1;
        }
        cursor += nchars;
        e->scan = scan;

    }
    printval("date %i/%i/%i\n", e->date_year, e->date_month, e->date_day);
    printval("scan %i\n", e->scan);

    if (sscanf(cursor, "%f|%f|%n", &e->glon, &e->glat, &nchars) != 2) { // [deg], [deg]
        fprintf(stderr, "Failed to parse 'glon/glat' entry in 2MASS line.\n");
        return -1;
    }
    cursor += nchars;
    printval("glon %g, glat %g.\n", e->glon, e->glat);

    if (sscanf(cursor, "%f|%lf|%n", &e->x_scan, &e->jdate, &nchars) != 2) {
        fprintf(stderr, "Failed to parse 'x_scan/jdate' entry in 2MASS line.\n");
        return -1;
    }
    cursor += nchars;
    e->x_scan = arcsec2deg(e->x_scan); // [arcsec : deg]
    printval("x_scan %g, jdate %g.\n", e->x_scan, e->jdate); // [day]

    for (i=0; i<9; i++) {
        char* names[] = {
            "j_psfchi", "h_psfchi", "k_psfchi",
            "j_m_stdap", "j_msig_stdap",
            "h_m_stdap", "h_msig_stdap",
            "k_m_stdap", "k_msig_stdap" };
        // all [mag] or unitless
        float* dests[] = {
            &e->j_psfchi, &e->h_psfchi, &e->k_psfchi,
            &e->j_m_stdap, &e->j_msig_stdap,
            &e->h_m_stdap, &e->h_msig_stdap,
            &e->k_m_stdap, &e->k_msig_stdap, };
        if (parse_null(&cursor, dests[i])) {
            fprintf(stderr, "Failed to parse \"%s\" entry in 2MASS line.\n", names[i]);
            return -1;
        }
        printval("%s: %g\n", names[i], *dests[i]);
    }

    {
        int dist_ns;
        int dist_ew;
        char ns;
        char ew;
        if (sscanf(cursor, "%i|%i|%c%c|%n", &dist_ns, &dist_ew, &ns, &ew, &nchars) != 4) {
            fprintf(stderr, "Failed to parse 'dist_edge_ns/dest_edge_ew/dist_edge_flag' entries in 2MASS line.\n");
            return -1;
        }
        cursor += nchars;
        e->dist_edge_ns = arcsec2deg(dist_ns); // [arcsec : deg]
        e->dist_edge_ew = arcsec2deg(dist_ew); // [arcsec : deg]
        switch (ns) {
        case 'n':
            e->dist_flag_ns = TRUE;
            break;
        case 's':
            e->dist_flag_ns = FALSE;
            break;
        default:
            fprintf(stderr, "Failed to parse 'dist_edge_flag' entry in 2MASS line.\n");
            return -1;
        }
        switch (ew) {
        case 'e':
            e->dist_flag_ew = TRUE;
            break;
        case 'w':
            e->dist_flag_ew = FALSE;
            break;
        default:
            fprintf(stderr, "Failed to parse 'dist_edge_flag' entry in 2MASS line.\n");
            return -1;
        }
    }
    printval("dist_ns %f %c, dest_ew %f %c.\n", e->dist_edge_ns, (e->dist_flag_ns ? 'N' : 'S'),
             e->dist_edge_ew, (e->dist_flag_ew ? 'E' : 'W'));

    if ((*cursor < '0') || (*cursor > '9')) {
        fprintf(stderr, "Failed to parse 'dup_src' entry in 2MASS line.\n");
        fprintf(stderr, "line: \"%s\"\n", line);
        fprintf(stderr, "cursor: \"%s\"\n", cursor);
        return -1;
    }
    e->dup_src = *cursor - '0';
    cursor++;
    printval("dup_src %i\n", e->dup_src);

    ensure(*cursor, "dup_src");
    cursor++;

    if ((*cursor == '1') || (*cursor == '0')) {
        e->use_src = (anbool)(*cursor - '0');
        cursor++;
    } else {
        fprintf(stderr, "Failed to parse 'use_src' entry in 2MASS line.\n");
        return -1;
    }
    printval("use_src %i\n", e->use_src);

    ensure(*cursor, "use_src");
    cursor++;

    switch (*cursor) {
    case '0':
        e->association = TWOMASS_ASSOCIATION_NONE;
        break;
    case 'T':
        e->association = TWOMASS_ASSOCIATION_TYCHO;
        break;
    case 'U':
        e->association = TWOMASS_ASSOCIATION_USNOA2;
        break;
    }
    cursor++;

    printval("association %i\n", e->association);

    ensure(*cursor, "association");
    cursor++;

    if (parse_null(&cursor, &val2) || // dist_opt
        parse_null(&cursor, &val3) || // phi_opt
        parse_null(&cursor, &e->b_m_opt) ||
        parse_null(&cursor, &e->vr_m_opt)) {
        fprintf(stderr, "Failed to parse 'dist_opt/phi_opt/b_m_opt/vr_m_opt' entries in 2MASS line.\n");
        return -1;
    }
    if (twomass_is_null_float(val2))
        e->dist_opt = TWOMASS_NULL;
    else
        e->dist_opt = arcsec2deg(val2); // [arcsec : deg]

    if (twomass_is_null_float(val3))
        e->phi_opt = TWOMASS_NULL;
    else
        e->phi_opt = val3;

    printval("dist_opt %g, phi_opt %g, b_m_opt %g, vr_m_opt %g.\n", e->dist_opt, e->phi_opt, e->b_m_opt, e->vr_m_opt);

    if ((*cursor >= '0') && (*cursor <= '9')) {
        e->nopt_mchs = *cursor - '0';
        cursor++;
    } else {
        fprintf(stderr, "Failed to parse 'nopt_mchs' entry in 2MASS line.\n");
        return -1;
    }
    cursor++;
    printval("nopt_matches %i\n", e->nopt_mchs);

    if (!strncmp(cursor, "\\N|", 3)) {
        e->xsc_key = TWOMASS_KEY_NULL;
        cursor += 3;
    } else {
        int ival;
        if (sscanf(cursor, "%i|%n", &ival, &nchars) != 1) {
            fprintf(stderr, "Failed to parse 'xsc_key' entry in 2MASS line.\n");
            return -1;
        }
        cursor += nchars;
    }

    printval("XSC key %i\n", e->xsc_key);

    {
        int scan, coadd_key, coadd;
        if (sscanf(cursor, "%i|%i|%i%n", &scan, &coadd_key, &coadd, &nchars) != 3) {
            fprintf(stderr, "Failed to parse 'scan/coadd_key/coadd' entries in 2MASS line.\n");
            return -1;
        }
        e->scan_key = scan;
        e->coadd_key = coadd_key;
        e->coadd = coadd;
        cursor += nchars;
    }

    printval("scan_key %i, coadd_key %i, coadd %i.\n", e->scan_key, e->coadd_key, e->coadd);

    return 0;
}

