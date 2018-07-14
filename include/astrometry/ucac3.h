/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#ifndef UCAC3_H
#define UCAC3_H

#include <stdint.h>

#define UCAC3_RECORD_SIZE 84

struct ucac3_entry {
    // (in brackets are the name, format, and units in the UCAC3 data files)
    // [degrees] (ra, I4: mas)
    double ra;
    // [degrees] (spd, I4: mas)
    double dec;

    // [degrees] (sigra, I2, mas)
    // error in RA*cos(Dec)
    float sigma_ra;
    // [degrees] (sigdc, I2, mas)
    float sigma_dec;

    // fit model mag
    // [mag] (im1, I2: millimag)
    // value 18.000 is used to flag errors
    float mag;

    // aperture mag
    // [mag] (im2, I2: millimag)
    // value 18.000 is used to flag errors
    float apmag;

    // [mag] (sigmag, I2: millimag)
    float mag_err;

    // (objt, I1)
    // -2 = warning: object could be from possible streak
    // -1 = warning: object is near overexposed star
    // 0 = good star
    // 1 = good star (data copied from another entry)
    // 2 = warning: contains at least 1 overexposed image
    // 3 = warning: all images are overexposed or "bad"
    int8_t objtype;

    // (dsf, I1)
    // 0 = single star
    // 1 = primary of pair with unreal secondary = single
    // 2 = forced separation, on same frame
    // 3 = blended image, some CCD frames show single star, some double
    // 4 = forced separation, 2 objects on same frame number
    // 5 = primary   component of real double
    // 6 = secondary component of real double
    // 7 = other "odd" case  
    int8_t doublestar;

    // (na1: I1)
    // total # of CCD images of this star
    uint8_t navail;

    // (nu1: I1)
    // # of CCD images used for this star
    uint8_t nused;

    // (us1: I1)
    // # catalogs (epochs) used for proper motions
    uint8_t npm;

    // (cn1: I1)
    // total numb. catalogs (epochs) initial match
    uint8_t nmatch;

    // Central epoch for mean RA/Dec
    // [yr] (cepra/cepdc, I2, 0.01 yr - 1900)
    float epoch_ra;
    float epoch_dec;

    // Proper motion in RA*cos(Dec), Dec
    // [arcsec/yr] (pmrac/pmdc, I4, 0.1 mas/yr)
    float pm_ra;
    float pm_dec;

    // [arcsec/pr] (sigpmr/sigpmd, I2, 0.1 mas/yr)
    float sigma_pm_ra;
    float sigma_pm_dec;

    // 2MASS pts_key star identifier
    // (id2m, I4)
    uint32_t twomass_id;

    // 2MASS J mag
    // (jmag, I2, millimag)
    float jmag;

    // 2MASS H mag
    // (hmag, I2, millimag)
    float hmag;

    // 2MASS K_s mag
    // (kmag, I2, millimag)
    float kmag;

    // e2mpho I*1 * 3         2MASS error photom. (1/100 mag)
    float jmag_err;
    float hmag_err;
    float kmag_err;

    // icqflg I*1 * 3         2MASS cc_flg*10 + phot.qual.flag
    /*
     (cc_flg*10 + ph_qual) consisting of the contamination flag (0 to 5) 
     and the photometric quality flag (0 to 8).  

     0 =  cc_flg  2MASS 0, no artifacts or contamination
     1 =  cc_flg  2MASS p, source may be contaminated by a latent image
     2 =  cc_flg  2MASS c, photometric confusion
     3 =  cc_flg  2MASS d, diffraction spike confusion
     4 =  cc_flg  2MASS s, electronic stripe
     5 =  cc_flg  2MASS b, bandmerge confusion

     0 =  no ph_qual flag
     1 =  ph_qual 2MASS X, no valid brightness estimate
     2 =  ph_qual 2MASS U, upper limit on magnitude
     3 =  ph_qual 2MASS F, no reliable estimate of the photometric error
     4 =  ph_qual 2MASS E, goodness-of-fit quality of profile-fit poor
     5 =  ph_qual 2MASS A, valid measurement, [jhk]snr>10 AND [jhk]cmsig<0.10857
     6 =  ph_qual 2MASS B, valid measurement, [jhk]snr> 7 AND [jhk]cmsig<0.15510
     7 =  ph_qual 2MASS C, valid measurement, [jhk]snr> 5 AND [jhk]cmsig<0.21714
     8 =  ph_qual 2MASS D, valid measurement, no [jhk]snr OR [jhk]cmsig req.
	 
     For example icqflg = 05 is decoded to be cc_flg=0, and ph_qual=5, meaning
     no artifacts or contamination from cc_flg and 2MASS qual flag = "A" .
     */
    uint8_t twomass_jflags;
    uint8_t twomass_hflags;
    uint8_t twomass_kflags;

    // SuperCosmos Bmag / R2mag / Imag
    // (smB / smR2 / smI, I2 millimag)
    float bmag;
    float r2mag;
    float imag;

    // clbl   I*1             SC star/galaxy classif./quality flag
    /*
     clbl is a combination of the SuperCOSMOS "meanclass" and
     modified "blend" flag (meanclass + 10*blend) originally for each
     magnitude (B, R1, R2, I).  The flag provided here is the maximum
     value over all bands for each of the 2 flags. The SuperCOSMOS mean 
     class flag is an estimate of image class based on unit-weighted mean 
     of individual classes from (B, R1, R2, I).  The SuperCOSMOS modified 
     blend flag indicates if blending is detected.
	 
     The "meanclass" has the following meaning from SuperCOSMOS:
     1 = galaxy
     2 = star
     3 = unclassifiable
     4 = noise

     The modified "blend" flag has the following meaning:
     0 = no blending
     1 = possible blending detected
     */
    uint8_t clbl;

    // SuperCosmos quality flag Bmag/R2mag/Imag
    // (qfB/qfR2/qfI, I1)
    /*
     This is a modified quality flag from the "qualB", "qualR2",
     and "qualI" quality flag from SuperCOSMOS, which gives an indication
     of the quality of the image from the three bands (B, R2, I).  The
     modified quality flag qfB, qfR2, and qfI have the following meaning:

     -1 = qual blank    in SuperCOSMOS, no flag given
     0 = qual zero     in SuperCOSMOS, no problems detected
     1 = qual < 128    in SuperCOSMOS, reliable image
     2 = qual < 65535  in SuperCOSMOS, problems detected
     3 = qual >= 65535 in SuperCOSMOS, spurious detection
     */
    uint8_t bquality;
    uint8_t r2quality;
    uint8_t iquality;

    // mmf flag for 10 major catalogs matched
    // (catflg, I * 10)
    /*
     The catflg provides reference to 10 major catalogs used
     in the computation of the proper motions and catalog matching.
     Each of the 10 numbers range from 0 to 6 and are the "mmf" 
     (multiple match flag) with respect to each of the 10 following
     catalogs:

     Hip, Tycho, AC2000, AGK2B, AGK2H, ZA, BY, Lick, SC, SPM
     catflg 1     2       3      4      5     6   7   8     9   10

     The value for each byte, the mmf flag, has the following meaning:

     0 = star not matched with this catalog
     1 = unique match,  not involving a double star
     2 = closest match, not involving a double, likely o.k. 
     3 = unique match,  and involving a double star
     4 = closest match, and involving a double, likely o.k.
     5 = maybe o.k. smallest sep. match in both directions
     6 = same as 5, but involving a double star
     */
    uint8_t matchflags[10];

    // Yale SPM object type (g-flag)
    // (g1, I1)
    /*
     The g-flag from the Yale San Juan first epoch Southern
     Proper Motion data (YSJ1, SPM) has the following meaning:

     0 = no info
     1 = matched with 2MASS extended source list
     2 = LEDA  galaxy
     3 = known QSO
     */
    uint8_t yale_gflag;
    // Yale SPM input cat.  (c-flag)
    // (c1, I1)
    /*
     The c-flag from the Yale San Juan first epoch Southern
     Proper Motion data (YSJ1, SPM) indicates which input catalog
     has been used to identify stars for pipeline processing:
	 
     1 = Hipparcos
     2 = Tycho2
     3 = UCAC2
     4 = 2MASS psc
     5 = 2MASS xsc (extended sources, largely (but not all!) galaxies)
     6 = LEDA  (confirmed galaxies, Paturel et al. 2005)
     7 = QSO   (Veron-Cetty & Veron 2006)
     */
    uint8_t yale_cflag;

    // LEDA galaxy match flag
    // (leda, I1)
    /*
     This flag is either 0 (no match) or contains the log of
     the apparent total diameter for I-band (object size) information
     copied from the LEDA catalog (galaxies).  A size value of less
     than 1 has been rounded up to 1.
     */
    uint8_t leda_flag;

    // 2MASS extend.source flag
    // (x2m, I1)
    /*
     This flag is either 0 (no match) or contains the length of
     the semi-major axis of the fiducial ellipse at the K-band 
     (object size) information copied from the 2MASS extended source
     catalog.  If the size is larger than 127, the flag was set to 127.
     */
    uint8_t twomass_extsource_flag;

    // MPOS star number; identifies HPM stars
    // (rn, I4)
    /*
     MPOS running star numbers over 140 million indicate high
     proper motion stars which were identified in UCAC pixel data from
     matches with known HPM stars.  The position given for those HPM
     stars is the unweighted mean of the CCD observations and the
     proper motion is copied from the literature catalog.
     */
    uint32_t mpos;
};

typedef struct ucac3_entry ucac3_entry;

int ucac3_parse_entry(ucac3_entry* entry, const void* encoded);

#endif
