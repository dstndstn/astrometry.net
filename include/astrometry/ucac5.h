/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

// Author: Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill

#ifndef UCAC5_H
#define UCAC5_H

#include <stdint.h>

#define UCAC5_RECORD_SIZE 52

struct ucac5_entry {
    // (in brackets are the description, format, and units in the UCAC5 data files)

    // (Gaia source ID, I8)
    int64_t srcid;

    // [degrees] (calculated RA at current epoch)
    double ra;

    // [degrees] (calculated Dec at current epoch)
    double dec;

    // [degrees] (Gaia DR1 RA at epoch 2015.0, I4: mas)
    double rag;

    // [degrees] (Gaia DR1 Dec at epoch 2015.0, I4: mas)
    double dcg;

    // [degrees] (Gaia DR1 position error RA at epoch 2015.0, I2: 0.1 mas)
    float erg;

    // [degrees] (Gaia DR1 position error Dec at epoch 2015.0, I2: 0.1 mas)
    float edg;

    // (1 = TGAS, 2 = other UCAC-Gaia star, 3 = other NOMAD, I1)
    uint8_t flg;

    // (number of images used for UCAC mean position, I1)
    uint8_t nu;

    // [yr] (mean UCAC epoch, I2: myr after 1997.0)
    float epu;

    // [degrees] (mean UCAC RA at epu epoch on Gaia reference frame, I4: mas)
    double ira;

    // [degrees] (mean UCAC Dec at epu epoch on Gaia reference frame, I4: mas)
    double idc;

    // [degrees/yr] (proper motion RA*cosDec (UCAC-Gaia), I2: 0.1 mas/yr)
    float pmur;

    // [degrees/yr] (proper motion Dec (UCAC-Gaia), I2: 0.1 mas/yr)
    float pmud;

    // [degrees/yr] (formal error of UCAC-Gaia proper motion RA*cosDec, I2: 0.1 mas/yr)
    float pmer;

    // [degrees/yr] (formal error of UCAC-Gaia proper motion Dec, I2: 0.1 mas/yr)
    float pmed;

    // [mag] (Gaia DR1 G magnitude, I2: mmag)
    float gmag;

    // [mag] (mean UCAC model magnitude, I2: mmag)
    float umag;

    // [mag] (NOMAD photographic R mag, I2: mmag)
    float rmag;

    // [mag] (2MASS J magnitude, I2: mmag)
    float jmag;

    // [mag] (2MASS H magnitude, I2: mmag)
    float hmag;

    // [mag] (2MASS K magnitude, I2: mmag)
    float kmag;
};

typedef struct ucac5_entry ucac5_entry;

int ucac5_parse_entry(ucac5_entry* entry, const void* encoded, float epoch);

#endif
