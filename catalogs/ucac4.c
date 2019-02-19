/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdint.h>
#include <stdio.h>

#include "ucac4.h"
#include "an-endian.h"
#include "starutil.h"

static uint32_t grab_u32(void** v) {
    uint32_t uval = u32_letoh(*((uint32_t*)(*v)));
    *v = (((uint32_t*)(*v)) + 1);
    //((uint32_t*)(*v)) += 1;
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
    //((uint16_t*)(*v)) += 1;
    return uval;
}

static int16_t grab_i16(void** v) {
    int16_t val = (int16_t)u16_letoh(*((uint16_t*)(*v)));
    *v = (((int16_t*)(*v)) + 1);
    return val;
}

static int8_t grab_i8(void** v) {
    int8_t val = *((int8_t*)(*v));
    *v = (((int8_t*)(*v)) + 1);
    return val;
}

static uint8_t grab_u8(void** v) {
    uint8_t val = *((uint8_t*)(*v));
    *v = (((uint8_t*)(*v)) + 1);
    return val;
}

int ucac4_parse_entry(ucac4_entry* entry, const void* encoded) {
    //const uint32_t* udata = encoded;
    uint32_t uval;
    void* buf = (void*)encoded;

    // RESIST THE URGE TO RE-ORDER THESE, bonehead!

    uval = grab_u32(&buf);
    entry->ra = arcsec2deg(uval * 0.001);

    uval = grab_u32(&buf);
    entry->dec = arcsec2deg(uval * 0.001) - 90.0;

    entry->mag     = 0.001 * grab_i16(&buf);
    entry->apmag   = 0.001 * grab_i16(&buf);
    entry->mag_err = 0.01  * grab_u8(&buf);
	
    entry->objtype    = grab_u8(&buf);
    entry->doublestar = grab_u8(&buf);

    entry->sigma_ra  = arcsec2deg(0.001 * (grab_i8(&buf) + 128));
    entry->sigma_dec = arcsec2deg(0.001 * (grab_i8(&buf) + 128));

    entry->navail = grab_u8(&buf);
    entry->nused  = grab_u8(&buf);
    entry->nmatch = grab_u8(&buf);

    entry->epoch_ra  = 1900. + 0.01 * grab_u16(&buf);
    entry->epoch_dec = 1900. + 0.01 * grab_u16(&buf);

    entry->pm_rac = 1e-4 * grab_i16(&buf);
    entry->pm_dec = 1e-4 * grab_i16(&buf);

    entry->sigma_pm_ra  = 1e-4 * (grab_u8(&buf) + 128);
    entry->sigma_pm_dec = 1e-4 * (grab_u8(&buf) + 128);

    entry->twomass_id = grab_u32(&buf);
    entry->jmag = 0.001 * grab_i16(&buf);
    entry->hmag = 0.001 * grab_i16(&buf);
    entry->kmag = 0.001 * grab_i16(&buf);

    entry->twomass_jflags = grab_u8(&buf);
    entry->twomass_hflags = grab_u8(&buf);
    entry->twomass_kflags = grab_u8(&buf);

    entry->jmag_err = 0.01 * grab_u8(&buf);
    entry->hmag_err = 0.01 * grab_u8(&buf);
    entry->kmag_err = 0.01 * grab_u8(&buf);

    entry->Bmag  = 0.001 * grab_i16(&buf);
    entry->Vmag  = 0.001 * grab_i16(&buf);
    entry->gmag  = 0.001 * grab_i16(&buf);
    entry->rmag  = 0.001 * grab_i16(&buf);
    entry->imag  = 0.001 * grab_i16(&buf);

    entry->Bmag_err  = 0.01 * grab_u8(&buf);
    entry->Vmag_err  = 0.01 * grab_u8(&buf);
    entry->gmag_err  = 0.01 * grab_u8(&buf);
    entry->rmag_err  = 0.01 * grab_u8(&buf);
    entry->imag_err  = 0.01 * grab_u8(&buf);

    entry->yale_gc_flag = grab_u8(&buf);
    entry->catalog_flags = grab_i32(&buf);
    entry->leda_flag = grab_u8(&buf);
    entry->twomass_extsource_flag = grab_u8(&buf);

    entry->mpos = grab_u32(&buf);
    entry->ucac2_zone = grab_u16(&buf);
    entry->ucac2_number= grab_u32(&buf);

    /*	 printf("ra=%g, dec=%g\n", entry->ra, entry->dec);
     printf("mag=%g, apmag=%g\n", entry->mag, entry->apmag);
     printf("objt=%i\n", entry->objtype); */
    return 0;
}
