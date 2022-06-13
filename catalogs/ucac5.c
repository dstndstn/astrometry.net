/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

// Author: Vladimir Kouprianov, Skynet RTN, University of North Carolina at Chapel Hill

#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "ucac5.h"
#include "an-endian.h"
#include "starutil.h"

static int64_t grab_i64(void** v) {
    int64_t val = (int64_t)u64_letoh(*((uint64_t*)(*v)));
    *v = (((int64_t*)(*v)) + 1);
    return val;
}

static uint32_t grab_u32(void** v) {
    uint32_t uval = u32_letoh(*((uint32_t*)(*v)));
    *v = (((uint32_t*)(*v)) + 1);
    return uval;
}

static int32_t grab_i32(void** v) {
    int32_t val = (int32_t)u32_letoh(*((uint32_t*)(*v)));
    *v = (((int32_t*)(*v)) + 1);
    return val;
}

static uint16_t grab_u16(void** v) {
    uint16_t uval = u16_letoh(*((uint16_t*)(*v)));
    *v = (((uint16_t*)(*v)) + 1);
    return uval;
}

static int16_t grab_i16(void** v) {
    int16_t val = (int16_t)u16_letoh(*((uint16_t*)(*v)));
    *v = (((int16_t*)(*v)) + 1);
    return val;
}

static uint8_t grab_u8(void** v) {
    uint8_t val = *((uint8_t*)(*v));
    *v = (((uint8_t*)(*v)) + 1);
    return val;
}

int ucac5_parse_entry(ucac5_entry* entry, const void* encoded, float epoch) {
    //const uint32_t* udata = encoded;
    void* buf = (void*)encoded;

    // RESIST THE URGE TO RE-ORDER THESE, bonehead!

    entry->srcid = grab_i64(&buf);

    entry->rag = arcsec2deg(grab_u32(&buf) * 0.001);
    entry->dcg = arcsec2deg(grab_i32(&buf) * 0.001);

    entry->erg = arcsec2deg(grab_u16(&buf) * 0.0001);
    entry->edg = arcsec2deg(grab_u16(&buf) * 0.0001);

    entry->flg = grab_u8(&buf);
    entry->nu = grab_u8(&buf);

    entry->epu = grab_i16(&buf) * 0.001 + 1997.0;

    entry->ira = arcsec2deg(grab_u32(&buf) * 0.001);
    entry->idc = arcsec2deg(grab_i32(&buf) * 0.001);

    entry->pmur = arcsec2deg(grab_i16(&buf) * 0.0001);
    entry->pmud = arcsec2deg(grab_i16(&buf) * 0.0001);

    entry->pmer = arcsec2deg(grab_u16(&buf) * 0.0001);
    entry->pmed = arcsec2deg(grab_u16(&buf) * 0.0001);

    // Apply proper motions
    if (epoch) {
      float dt = epoch - entry->epu;
      double cosDec = cos(entry->idc*M_PI/180.0);
      double dra = entry->pmur*(cosDec ? dt/cosDec : dt);
      double ra = fmod(entry->ira + dra, 360.0);
      double dec = entry->idc + entry->pmud*dt;
      if (dec > 90.0) {
        dec = 180.0 - dec;
        ra = fmod(ra + 180.0, 360.0);
      }
      else if (dec < -90.0) {
        dec = -180.0 - dec;
        ra = fmod(ra + 180.0, 360.0);
      }
      entry->ra = ra;
      entry->dec = dec;
    }
    else {
      entry->ra = entry->ira;
      entry->dec = entry->idc;
    }

    entry->gmag = grab_i16(&buf) * 0.001;
    entry->umag = grab_i16(&buf) * 0.001;
    entry->rmag = grab_i16(&buf) * 0.001;
    entry->jmag = grab_i16(&buf) * 0.001;
    entry->hmag = grab_i16(&buf) * 0.001;
    entry->kmag = grab_i16(&buf) * 0.001;

    return 0;
}
